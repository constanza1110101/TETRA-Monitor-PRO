#!/usr/bin/env python3
# TETRA Monitor Pro - Herramienta avanzada de análisis para señales TETRA
# SOLO PARA USO EDUCATIVO, INVESTIGACIÓN Y LEGAL
# Requiere: Python 3.x, gr-osmosdr, numpy, matplotlib, sqlite3, pyaudio, scipy, tensorflow

import numpy as np
import matplotlib.pyplot as plt
from gnuradio import gr, blocks, filter, analog
import osmosdr
import time
import argparse
import json
import os
import sqlite3
import pyaudio
import threading
import hashlib
import logging
import socket
import requests
import ipaddress
import pandas as pd
import folium
from datetime import datetime
from scipy import signal
from scipy.io import wavfile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import tensorflow as tf
import webbrowser
import csv
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue
import gpsd
import serial
import glob
from http.server import HTTPServer, BaseHTTPRequestHandler
import qrcode
from io import BytesIO
from PIL import Image, ImageTk

class TetraMonitorPro:
    def __init__(self, freq=395e6, sample_rate=2e6, gain=30, device="airspy"):
        self.freq = freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.device = device
        self.running = False
        self.recording = False
        self.base_directory = "tetra_data"
        self.log_directory = os.path.join(self.base_directory, "logs")
        self.audio_directory = os.path.join(self.base_directory, "audio")
        self.spectrum_directory = os.path.join(self.base_directory, "spectrum")
        self.export_directory = os.path.join(self.base_directory, "exports")
        self.ml_models_directory = os.path.join(self.base_directory, "ml_models")
        self.protocols_directory = os.path.join(self.base_directory, "protocols")
        self.map_directory = os.path.join(self.base_directory, "maps")
        self.reports_directory = os.path.join(self.base_directory, "reports")
        self.keys_directory = os.path.join(self.base_directory, "keys")
        
        # Crear directorios si no existen
        for directory in [self.log_directory, self.audio_directory, self.spectrum_directory, 
                         self.export_directory, self.ml_models_directory, self.protocols_directory,
                         self.map_directory, self.reports_directory, self.keys_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Configurar logging
        logging.basicConfig(
            filename=os.path.join(self.log_directory, f'tetra_monitor_{time.strftime("%Y%m%d")}.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('tetra_monitor')
        
        # Cola para comunicación entre hilos
        self.message_queue = queue.Queue()
        
        # Inicializar base de datos
        self._init_database()
        
        # Inicializar sistema GPS si está disponible
        self.gps_available = False
        try:
            gpsd.connect()
            self.gps_available = True
            self.logger.info("GPS conectado correctamente")
        except:
            self.logger.warning("GPS no disponible")
            # Intentar conexión serial a GPS
            self._try_serial_gps()
        
        # Cargar configuración de red
        self.network_config = self._load_network_config()
        
        # Inicializar modelos de ML
        self.ml_models = {}
        self._load_ml_models()
        
        # Cargar base de datos de operadores TETRA
        self.operators_db = self._load_operators_db()
        
        # Cargar base de datos de frecuencias conocidas
        self.known_frequencies = self._load_known_frequencies()
        
        # Inicializar servidor web
        self.web_server_running = False
        self.web_server_thread = None
        
        # Inicializar sistema de alertas
        self.alert_system = {
            "enabled": True,
            "thresholds": {
                "power": -50,  # dBm
                "bandwidth": 50,  # kHz
                "jamming": True
            },
            "notifications": {
                "email": False,
                "sound": True,
                "popup": True
            }
        }
        
        # Inicializar sistema de análisis de patrones
        self.pattern_analyzer = {
            "enabled": True,
            "history_size": 100,
            "correlation_threshold": 0.8
        }
        
        # Inicializar sistema de decodificación
        self.decoder = {
            "enabled": True,
            "modes": {
                "tetra": True,
                "dmr": True,
                "p25": False,
                "nxdn": False
            }
        }
    
    def _try_serial_gps(self):
        """Intenta conectar con GPS por puerto serial"""
        # Buscar puertos seriales disponibles
        ports = glob.glob('/dev/tty[A-Za-z]*') if os.name != 'nt' else \
                ['COM%s' % (i + 1) for i in range(256)]
        
        for port in ports:
            try:
                s = serial.Serial(port, 9600, timeout=1)
                s.close()
                self.gps_port = port
                self.gps_available = True
                self.logger.info(f"GPS detectado en puerto serial {port}")
                return
            except (OSError, serial.SerialException):
                pass
    
    def _load_network_config(self):
        """Carga configuración de red desde archivo JSON"""
        config_file = os.path.join(self.base_directory, "network_config.json")
        
        # Configuración por defecto
        default_config = {
            "remote_server": "",
            "api_key": "",
            "enable_remote_logging": False,
            "enable_remote_storage": False,
            "enable_realtime_sharing": False,
            "trusted_peers": [],
            "network_scan_enabled": False,
            "web_interface": {
                "enabled": False,
                "port": 8080,
                "require_auth": True,
                "username": "admin",
                "password_hash": hashlib.sha256("admin".encode()).hexdigest()
            }
        }
        
        # Crear archivo de configuración si no existe
        if not os.path.exists(config_file):
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
        
        # Cargar configuración existente
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except:
            self.logger.error("Error al cargar configuración de red")
            return default_config
    
    def _load_ml_models(self):
        """Carga modelos de machine learning preentrenados"""
        # Modelo para clasificación de señales
        model_path = os.path.join(self.ml_models_directory, "signal_classifier.h5")
        if os.path.exists(model_path):
            try:
                self.ml_models["signal_classifier"] = tf.keras.models.load_model(model_path)
                self.logger.info("Modelo de clasificación de señales cargado correctamente")
            except:
                self.logger.warning("Error al cargar modelo de clasificación de señales")
        
        # Modelo para detección de anomalías
        model_path = os.path.join(self.ml_models_directory, "anomaly_detector.h5")
        if os.path.exists(model_path):
            try:
                self.ml_models["anomaly_detector"] = tf.keras.models.load_model(model_path)
                self.logger.info("Modelo de detección de anomalías cargado correctamente")
            except:
                self.logger.warning("Error al cargar modelo de detección de anomalías")
        
        # Si no existen modelos, crear modelos básicos
        if not self.ml_models:
            self._create_basic_ml_models()
    
    def _create_basic_ml_models(self):
        """Crea modelos básicos de ML para clasificación y detección de anomalías"""
        try:
            # Modelo de clasificación básico
            classifier = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(1024,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(6, activation='softmax')  # TETRA, DMR, P25, NXDN, Analog, Unknown
            ])
            classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Modelo de detección de anomalías (autoencoder)
            encoder = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(1024,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu')
            ])
            
            decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(1024, activation='sigmoid')
            ])
            
            autoencoder = tf.keras.Sequential([encoder, decoder])
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Guardar modelos
            classifier_path = os.path.join(self.ml_models_directory, "signal_classifier.h5")
            autoencoder_path = os.path.join(self.ml_models_directory, "anomaly_detector.h5")
            
            classifier.save(classifier_path)
            autoencoder.save(autoencoder_path)
            
            self.ml_models["signal_classifier"] = classifier
            self.ml_models["anomaly_detector"] = autoencoder
            
            self.logger.info("Modelos básicos de ML creados correctamente")
        except Exception as e:
            self.logger.error(f"Error al crear modelos básicos de ML: {e}")
    
    def _load_operators_db(self):
        """Carga base de datos de operadores TETRA"""
        db_file = os.path.join(self.base_directory, "operators_db.json")
        
        # Base de datos por defecto
        default_db = {
            "operators": [
                {
                    "id": 1,
                    "name": "SIRDEE",
                    "country": "España",
                    "mcc": 214,
                    "mnc": 1,
                    "freq_range": [380e6, 400e6],
                    "channels": [380.0125e6, 380.0250e6, 380.0375e6]  # Ejemplo
                },
                {
                    "id": 2,
                    "name": "BOSNet",
                    "country": "Alemania",
                    "mcc": 262,
                    "mnc": 1,
                    "freq_range": [380e6, 400e6],
                    "channels": [385.0125e6, 385.0250e6, 385.0375e6]  # Ejemplo
                },
                {
                    "id": 3,
                    "name": "Airwave",
                    "country": "Reino Unido",
                    "mcc": 234,
                    "mnc": 1,
                    "freq_range": [380e6, 400e6],
                    "channels": [390.0125e6, 390.0250e6, 390.0375e6]  # Ejemplo
                }
            ]
        }
        
        # Crear archivo si no existe
        if not os.path.exists(db_file):
            with open(db_file, 'w') as f:
                json.dump(default_db, f, indent=2)
            return default_db
        
        # Cargar base de datos existente
        try:
            with open(db_file, 'r') as f:
                db = json.load(f)
            return db
        except:
            self.logger.error("Error al cargar base de datos de operadores")
            return default_db
    
    def _load_known_frequencies(self):
        """Carga base de datos de frecuencias conocidas"""
        db_file = os.path.join(self.base_directory, "known_frequencies.json")
        
        # Base de datos por defecto
        default_db = {
            "tetra": {
                "380-400": "Bandas TETRA para seguridad pública en Europa",
                "410-430": "Bandas TETRA para uso comercial",
                "450-470": "Bandas TETRA para uso comercial en algunos países"
            },
            "dmr": {
                "136-174": "VHF para uso comercial",
                "403-470": "UHF para uso comercial",
                "450-520": "UHF para uso comercial en algunos países"
            },
            "p25": {
                "136-174": "VHF para seguridad pública",
                "380-400": "UHF para seguridad pública",
                "746-806": "700/800 MHz para seguridad pública en EE.UU."
            }
        }
        
        # Crear archivo si no existe
        if not os.path.exists(db_file):
            with open(db_file, 'w') as f:
                json.dump(default_db, f, indent=2)
            return default_db
        
        # Cargar base de datos existente
        try:
            with open(db_file, 'r') as f:
                db = json.load(f)
            return db
        except:
            self.logger.error("Error al cargar base de datos de frecuencias conocidas")
            return default_db
    
    def _init_database(self):
        """Inicializa la base de datos SQLite para almacenar información de señales"""
        db_path = os.path.join(self.base_directory, "tetra_signals.db")
        self.conn = sqlite3.connect(db_path)
        cursor = self.conn.cursor()
        
        # Crear tablas si no existen
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frequency REAL,
            power REAL,
            bandwidth REAL,
            timestamp TEXT,
            location TEXT,
            classification TEXT,
            confidence REAL,
            notes TEXT,
            hash TEXT,
            mcc INTEGER,
            mnc INTEGER,
            operator_id INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            filename TEXT,
            duration REAL,
            timestamp TEXT,
            FOREIGN KEY (signal_id) REFERENCES signals (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            timestamp TEXT,
            FOREIGN KEY (signal_id) REFERENCES signals (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS protocol_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            protocol_type TEXT,
            raw_data BLOB,
            decoded_data TEXT,
            timestamp TEXT,
            FOREIGN KEY (signal_id) REFERENCES signals (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS networks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            frequency_range TEXT,
            location_area TEXT,
            first_seen TEXT,
            last_seen TEXT,
            notes TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            description TEXT,
            frequency REAL,
            power REAL,
            timestamp TEXT,
            acknowledged BOOLEAN
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS decryption_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol TEXT,
            key_type TEXT,
            key_id TEXT,
            key_data BLOB,
            description TEXT,
            timestamp TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            device TEXT,
            sample_rate REAL,
            location TEXT,
            notes TEXT
        )
        ''')
        
        self.conn.commit()
        self.logger.info("Base de datos inicializada")
    
    def setup_receiver(self):
        """Configura el receptor SDR"""
        try:
            # Configurar fuente SDR
            self.sdr = osmosdr.source(args="device=" + self.device)
            self.sdr.set_sample_rate(self.sample_rate)
            self.sdr.set_center_freq(self.freq)
            self.sdr.set_freq_corr(0)
            self.sdr.set_gain(self.gain)
            
            # Registrar sesión
            self._register_session()
            
            self.logger.info(f"Receptor configurado: {self.device} en {self.freq/1e6} MHz")
            print(f"Receptor configurado: {self.device} en {self.freq/1e6} MHz")
            return True
        except Exception as e:
            self.logger.error(f"Error al configurar el receptor: {e}")
            print(f"Error al configurar el receptor: {e}")
            return False
    
    def _register_session(self):
        """Registra la sesión actual en la base de datos"""
        location = "Desconocida"
        if self.gps_available:
            try:
                packet = gpsd.get_current()
                location = f"{packet.lat},{packet.lon}"
            except:
                pass
        
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO sessions (start_time, device, sample_rate, location, notes)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.device,
            self.sample_rate,
            location,
            f"Frecuencia inicial: {self.freq/1e6} MHz"
        ))
        self.conn.commit()
        self.session_id = cursor.lastrowid
    
    def _close_session(self):
        """Cierra la sesión actual en la base de datos"""
        if hasattr(self, 'session_id'):
            cursor = self.conn.cursor()
            cursor.execute('''
            UPDATE sessions SET end_time = ? WHERE id = ?
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.session_id
            ))
            self.conn.commit()
    
    def scan_tetra_band(self, start_freq=380e6, end_freq=400e6, step=25e3, advanced_mode=False):
        """Escanea la banda TETRA en busca de actividad y analiza las señales encontradas"""
        self.logger.info(f"Escaneando banda TETRA ({start_freq/1e6}-{end_freq/1e6} MHz)...")
        print(f"Escaneando banda TETRA ({start_freq/1e6}-{end_freq/1e6} MHz)...")
        
        active_channels = []
        current_freq = start_freq
        
        # Crear gráfico en tiempo real
        plt.figure(figsize=(12, 8))
        plt.ion()
        
        # Obtener ubicación GPS si está disponible
        location_info = self._get_gps_location()
        
        # Crear mapa si GPS está disponible
        if location_info:
            map_center = [location_info["latitude"], location_info["longitude"]]
            signal_map = folium.Map(location=map_center, zoom_start=12)
            folium.Marker(
                location=map_center,
                popup="Posición actual",
                icon=folium.Icon(color='blue')
            ).add_to(signal_map)
        else:
            signal_map = None
        
        try:
            while current_freq <= end_freq:
                self.sdr.set_center_freq(current_freq)
                time.sleep(0.1)  # Tiempo para estabilizar
                
                # Obtener muestras y calcular potencia
                samples = self.sdr.get_samples(8192)
                power = np.mean(np.abs(samples)**2)
                
                # Análisis espectral para estimar ancho de banda
                spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2
                freq_range = np.linspace(current_freq - self.sample_rate/2, 
                                        current_freq + self.sample_rate/2, 
                                        len(spectrum)) / 1e6
                
                # Umbral de detección
                if power > 0.01:  # Ajustar según sensibilidad deseada
                    # Estimar ancho de banda
                    bandwidth = self._estimate_bandwidth(spectrum, freq_range)
                    
                    # Clasificar señal si el modelo está disponible
                    classification = "Unknown"
                    confidence = 0.0
                    
                    if "signal_classifier" in self.ml_models and advanced_mode:
                        classification, confidence = self._classify_signal(samples, spectrum)
                    
                    # Identificar operador
                    operator_id, mcc, mnc = self._identify_operator(current_freq)
                    
                    # Crear hash único para la señal
                    signal_hash = hashlib.md5(f"{current_freq}_{time.time()}".encode()).hexdigest()
                    
                    channel_info = {
                        "frequency": current_freq,
                        "power": float(power),
                        "bandwidth": bandwidth,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "location": json.dumps(location_info) if location_info else None,
                        "classification": classification,
                        "confidence": confidence,
                        "hash": signal_hash,
                        "operator_id": operator_id,
                        "mcc": mcc,
                        "mnc": mnc
                    }
                    
                    # Guardar en base de datos
                    signal_id = self._save_signal_to_db(channel_info)
                    
                    # Guardar localización si está disponible
                    if location_info:
                        self._save_location_to_db(signal_id, location_info)
                        
                        # Añadir al mapa
                        if signal_map:
                            popup_text = f"""
                            <b>Frecuencia:</b> {current_freq/1e6:.3f} MHz<br>
                            <b>Potencia:</b> {10*np.log10(power):.2f} dB<br>
                            <b>Ancho de banda:</b> {bandwidth:.1f} kHz<br>
                            <b>Tipo:</b> {classification}<br>
                            <b>Confianza:</b> {confidence:.2f}
                            """
                            
                            folium.CircleMarker(
                                location=[location_info["latitude"], location_info["longitude"]],
                                radius=5,
                                popup=folium.Popup(popup_text, max_width=300),
                                color='red' if classification == "TETRA" else 'green',
                                fill=True
                            ).add_to(signal_map)
                    
                    active_channels.append(channel_info)
                    self.logger.info(f"Actividad detectada en {current_freq/1e6:.3f} MHz (potencia: {power:.6f}, BW: {bandwidth:.1f} kHz, tipo: {classification})")
                    print(f"Actividad detectada en {current_freq/1e6:.3f} MHz (potencia: {power:.6f}, BW: {bandwidth:.1f} kHz, tipo: {classification})")
                    
                    # Visualizar
                    plt.clf()
                    plt.plot(freq_range, 10 * np.log10(spectrum))
                    plt.axvline(x=current_freq/1e6, color='r', linestyle='--')
                    plt.grid(True)
                    plt.xlabel('Frecuencia (MHz)')
                    plt.ylabel('Potencia (dB)')
                    plt.title(f'Señal detectada en {current_freq/1e6:.3f} MHz - {classification}')
                    plt.pause(0.5)
                    
                    # Análisis avanzado si está habilitado
                    if advanced_mode:
                        self._analyze_protocol(samples, current_freq, signal_id)
                        
                        # Detectar anomalías
                        if "anomaly_detector" in self.ml_models:
                            is_anomaly = self._detect_anomalies(samples, spectrum)
                            if is_anomaly:
                                self.logger.warning(f"ANOMALÍA detectada en {current_freq/1e6:.3f} MHz")
                                print(f"ANOMALÍA detectada en {current_freq/1e6:.3f} MHz")
                                
                                # Generar alerta
                                self._generate_alert("anomaly", f"Anomalía detectada en {current_freq/1e6:.3f} MHz", current_freq, power)
                
                current_freq += step
                
                # Actualizar ubicación GPS periódicamente
                if int(time.time()) % 30 == 0:
                    location_info = self._get_gps_location()
        
        except KeyboardInterrupt:
            self.logger.info("Escaneo interrumpido por el usuario")
            print("Escaneo interrumpido por el usuario")
        finally:
            plt.ioff()
            plt.close()
        
        # Guardar mapa si está disponible
        if signal_map:
            map_path = os.path.join(self.map_directory, f"signals_map_{time.strftime('%Y%m%d_%H%M%S')}.html")
            signal_map.save(map_path)
            self.logger.info(f"Mapa de señales guardado en {map_path}")
            print(f"Mapa de señales guardado en {map_path}")
            
            # Abrir mapa en navegador
            webbrowser.open('file://' + os.path.abspath(map_path))
        
        # Guardar resultados
        self._save_scan_results(active_channels)
        
        # Generar informe
        self._generate_scan_report(active_channels)
        
        return active_channels
    
    def _get_gps_location(self):
        """Obtiene la ubicación actual desde GPS"""
        if not self.gps_available:
            return None
        
        try:
            # Intentar con gpsd primero
            packet = gpsd.get_current()
            return {
                "latitude": packet.lat,
                "longitude": packet.lon,
                "altitude": packet.alt,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except:
            # Intentar con conexión serial
            if hasattr(self, 'gps_port'):
                try:
                    with serial.Serial(self.gps_port, 9600, timeout=1) as ser:
                        for _ in range(10):  # Intentar leer algunas líneas
                            line = ser.readline().decode('ascii', errors='replace').strip()
                            if line.startswith('$GPGGA'):  # Formato NMEA
                                parts = line.split(',')
                                if len(parts) >= 10 and parts[2] and parts[4]:
                                    lat = float(parts[2][:2]) + float(parts[2][2:]) / 60
                                    lon = float(parts[4][:3]) + float(parts[4][3:]) / 60
                                    if parts[3] == 'S': lat = -lat
                                    if parts[5] == 'W': lon = -lon
                                    alt = float(parts[9]) if parts[9] else 0
                                    
                                    return {
                                        "latitude": lat,
                                        "longitude": lon,
                                        "altitude": alt,
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                    }
                except:
                    self.logger.error("Error al leer GPS por puerto serial")
            
            return None
    
    def _save_location_to_db(self, signal_id, location_info):
        """Guarda información de localización en la base de datos"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO locations (signal_id, latitude, longitude, altitude, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            signal_id,
            location_info["latitude"],
            location_info["longitude"],
            location_info["altitude"],
            location_info["timestamp"]
        ))
        self.conn.commit()
    
    def _classify_signal(self, samples, spectrum):
        """Clasifica el tipo de señal usando modelo de machine learning"""
        try:
            # Preparar características para el modelo
            features = np.concatenate([
                np.abs(samples[:1000]),  # Amplitud de muestras
                10 * np.log10(spectrum[:1000]),  # Espectro en dB
                np.angle(samples[:1000])  # Fase
            ])
            
            # Normalizar
            features = (features - np.mean(features)) / np.std(features)
            
            # Redimensionar para el modelo
            features = features.reshape(1, -1)
            
            # Predicción
            prediction = self.ml_models["signal_classifier"].predict(features)
            
            # Clases posibles (ajustar según el modelo entrenado)
            classes = ["TETRA", "TETRAPOL", "DMR", "P25", "NXDN", "Unknown"]
            
            # Obtener clase y confianza
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            
              return classes[class_idx], confidence
        except:  Exception as e:
            self.logger.error(f"Error en clasificación de señal: {e}")
            return "Unknown", 0.0
    
    def _detect_anomalies(self, samples, spectrum):
        """Detecta anomalías en la señal usando modelo de machine learning"""
        try:
            # Preparar características para el modelo
            features = np.concatenate([
                np.abs(samples[:1000]),  # Amplitud de muestras
                10 * np.log10(spectrum[:1000]),  # Espectro en dB
            ])
            
            # Normalizar
            features = (features - np.mean(features)) / np.std(features)
            
            # Redimensionar para el modelo
            features = features.reshape(1, -1)
            
            # Predicción (modelo autoencoder)
            reconstructed = self.ml_models["anomaly_detector"].predict(features)
            
            # Calcular error de reconstrucción
            mse = np.mean(np.power(features - reconstructed, 2))
            
            # Umbral para considerar anomalía
            threshold = 0.15  # Ajustar según entrenamiento
            
            return mse > threshold
        except Exception as e:
            self.logger.error(f"Error en detección de anomalías: {e}")
            return False
    
    def _identify_operator(self, frequency):
        """Intenta identificar el operador basado en la frecuencia"""
        operator_id = None
        mcc = None
        mnc = None
        
        # Buscar en la base de datos de operadores
        for operator in self.operators_db["operators"]:
            freq_range = operator["freq_range"]
            if freq_range[0] <= frequency <= freq_range[1]:
                # Verificar si es un canal conocido
                if "channels" in operator and frequency in operator["channels"]:
                    operator_id = operator["id"]
                    mcc = operator["mcc"]
                    mnc = operator["mnc"]
                    break
                # Si no es un canal específico pero está en el rango
                elif abs(frequency - round(frequency / 25e3) * 25e3) < 1e3:  # Tolerancia de 1 kHz
                    operator_id = operator["id"]
                    mcc = operator["mcc"]
                    mnc = operator["mnc"]
                    break
        
        return operator_id, mcc, mnc
    
    def _analyze_protocol(self, samples, frequency, signal_id):
        """Analiza el protocolo de la señal y extrae información básica"""
        try:
            # Análisis básico de estructura TETRA
            # Nota: Esta es una implementación simplificada para fines educativos
            
            # Demodulación básica
            demod = np.angle(samples[1:] * np.conj(samples[:-1]))
            
            # Buscar patrones de sincronización
            correlation = np.correlate(demod, np.sin(np.linspace(0, 8*np.pi, 100)), mode='valid')
            sync_positions = signal.find_peaks(np.abs(correlation), height=0.5*np.max(np.abs(correlation)))[0]
            
            if len(sync_positions) > 0:
                # Determinar tipo de protocolo basado en frecuencia
                protocol_type = "Unknown"
                if 380e6 <= frequency <= 400e6:
                    protocol_type = "TETRA"
                elif 410e6 <= frequency <= 430e6:
                    protocol_type = "TETRA"
                elif 450e6 <= frequency <= 470e6:
                    protocol_type = "DMR/TETRA"
                
                # Estimar tasa de símbolos
                if len(sync_positions) > 1:
                    symbol_intervals = np.diff(sync_positions)
                    avg_interval = np.mean(symbol_intervals)
                    symbol_rate = self.sample_rate / avg_interval
                else:
                    symbol_rate = 0
                
                # Datos decodificados (simulados para fines educativos)
                decoded_data = {
                    "protocol": protocol_type,
                    "sync_positions": sync_positions.tolist(),
                    "pattern_count": len(sync_positions),
                    "estimated_symbol_rate": f"{symbol_rate:.2f} symbols/s",
                    "modulation": self._estimate_modulation(samples),
                    "encryption": self._check_encryption(demod)
                }
                
                # Guardar datos en la base de datos
                cursor = self.conn.cursor()
                cursor.execute('''
                INSERT INTO protocol_data (signal_id, protocol_type, raw_data, decoded_data, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    signal_id,
                    protocol_type,
                    sqlite3.Binary(samples.tobytes()),  # Datos crudos
                    json.dumps(decoded_data),
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ))
                self.conn.commit()
                
                self.logger.info(f"Análisis de protocolo completado: {protocol_type}, {len(sync_positions)} patrones detectados")
                
                # Si es TETRA y el decodificador está habilitado, intentar extraer más información
                if protocol_type == "TETRA" and self.decoder["enabled"] and self.decoder["modes"]["tetra"]:
                    self._decode_tetra_basic(demod, signal_id)
                
                return True
        except Exception as e:
            self.logger.error(f"Error en análisis de protocolo: {e}")
        
        return False
    
    def _estimate_modulation(self, samples):
        """Estima el tipo de modulación de la señal"""
        # Calcular estadísticas de la señal
        amplitude = np.abs(samples)
        phase = np.angle(samples)
        
        # Histograma de amplitud
        amp_hist, _ = np.histogram(amplitude, bins=50)
        amp_peaks = signal.find_peaks(amp_hist)[0]
        
        # Histograma de fase
        phase_hist, _ = np.histogram(phase, bins=50)
        phase_peaks = signal.find_peaks(phase_hist)[0]
        
        # Decisión basada en histogramas
        if len(amp_peaks) >= 2 and len(phase_peaks) <= 2:
            return "ASK/OOK"
        elif len(amp_peaks) <= 2 and len(phase_peaks) >= 4:
            return "PSK"
        elif len(amp_peaks) >= 2 and len(phase_peaks) >= 3:
            return "QAM"
        elif len(amp_peaks) <= 2 and len(phase_peaks) <= 2:
            # Calcular desviación de frecuencia
            freq_dev = np.std(np.diff(phase))
            if freq_dev > 0.1:
                return "FM"
            else:
                return "PM"
        else:
            return "Unknown"
    
    def _check_encryption(self, demod_signal):
        """Verifica si la señal parece estar encriptada"""
        # Calcular entropía de la señal demodulada
        hist, _ = np.histogram(demod_signal, bins=50, density=True)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Las señales encriptadas tienden a tener alta entropía
        if entropy > 4.0:
            return {
                "encrypted": True,
                "entropy": float(entropy),
                "type": "Posiblemente TEA (TETRA Encryption Algorithm)"
            }
        else:
            return {
                "encrypted": False,
                "entropy": float(entropy)
            }
    
    def _decode_tetra_basic(self, demod_signal, signal_id):
        """Realiza una decodificación básica de señales TETRA (para fines educativos)"""
        # Esta es una simulación simplificada para fines educativos
        # En un sistema real, la decodificación TETRA es mucho más compleja
        
        # Buscar patrones de sincronización
        sync_pattern = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        correlation = np.correlate(np.sign(demod_signal), sync_pattern, mode='valid')
        sync_positions = signal.find_peaks(correlation, height=0.7*len(sync_pattern))[0]
        
        if len(sync_positions) > 0:
            # Extraer información básica
            info = {
                "sync_positions": len(sync_positions),
                "burst_structure": "Posible trama TETRA detectada",
                "slots": min(len(sync_positions), 4),  # TETRA usa hasta 4 slots por trama
                "tdma_structure": "Posible estructura TDMA de 4 slots detectada"
            }
            
            # Simular extracción de MCC/MNC (esto es solo educativo)
            # En un sistema real, estos valores se extraerían de la cabecera de la trama
            cursor = self.conn.cursor()
            cursor.execute("SELECT mcc, mnc FROM signals WHERE id = ?", (signal_id,))
            result = cursor.fetchone()
            
            if result and result[0] and result[1]:
                info["mcc"] = result[0]
                info["mnc"] = result[1]
                info["network_info"] = f"Red TETRA MCC:{result[0]} MNC:{result[1]}"
            
            # Actualizar la información de protocolo
            cursor.execute('''
            SELECT id, decoded_data FROM protocol_data 
            WHERE signal_id = ? ORDER BY id DESC LIMIT 1
            ''', (signal_id,))
            
            protocol_result = cursor.fetchone()
            if protocol_result:
                protocol_id = protocol_result[0]
                decoded_data = json.loads(protocol_result[1])
                
                # Añadir nueva información
                decoded_data.update(info)
                
                # Actualizar en la base de datos
                cursor.execute('''
                UPDATE protocol_data SET decoded_data = ? WHERE id = ?
                ''', (json.dumps(decoded_data), protocol_id))
                
                self.conn.commit()
                self.logger.info("Información TETRA básica decodificada y guardada")
    
    def _estimate_bandwidth(self, spectrum, freq_range):
        """Estima el ancho de banda de una señal basado en su espectro"""
        # Convertir a dB
        spectrum_db = 10 * np.log10(spectrum)
        
        # Encontrar el pico
        max_idx = np.argmax(spectrum_db)
        peak_power = spectrum_db[max_idx]
        
        # Encontrar puntos a -3dB del pico
        threshold = peak_power - 3
        
        # Encontrar índices donde el espectro cruza el umbral
        above_threshold = spectrum_db > threshold
        
        # Encontrar el primer y último índice donde se cruza el umbral
        try:
            indices = np.where(above_threshold)[0]
            first_idx = indices[0]
            last_idx = indices[-1]
            
            # Calcular ancho de banda en kHz
            bandwidth = (freq_range[last_idx] - freq_range[first_idx]) * 1000
            return bandwidth
        except:
            return 25.0  # Valor predeterminado para TETRA
    
    def _save_signal_to_db(self, signal_info):
        """Guarda información de la señal en la base de datos y devuelve el ID"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO signals (frequency, power, bandwidth, timestamp, location, classification, confidence, hash, operator_id, mcc, mnc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_info["frequency"],
            signal_info["power"],
            signal_info.get("bandwidth", 0),
            signal_info["timestamp"],
            signal_info.get("location"),
            signal_info.get("classification", "Unknown"),
            signal_info.get("confidence", 0.0),
            signal_info["hash"],
            signal_info.get("operator_id"),
            signal_info.get("mcc"),
            signal_info.get("mnc")
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def _generate_alert(self, alert_type, description, frequency, power):
        """Genera una alerta en la base de datos"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO alerts (type, description, frequency, power, timestamp, acknowledged)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            alert_type,
            description,
            frequency,
            power,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            False
        ))
        self.conn.commit()
        
        self.logger.warning(f"ALERTA: {description}")
        print(f"\n¡ALERTA! {description}\n")
    
    def _save_scan_results(self, channels):
        """Guarda resultados del escaneo en archivo JSON"""
        filename = os.path.join(self.log_directory, 
                              f"tetra_scan_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, 'w') as f:
            json.dump(channels, f, indent=2)
        
        # También exportar como CSV
        csv_filename = os.path.join(self.export_directory, 
                                  f"tetra_scan_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Escribir encabezados
            writer.writerow(["Frecuencia (MHz)", "Potencia (dB)", "Ancho de banda (kHz)", 
                           "Timestamp", "Clasificación", "Confianza", "MCC", "MNC", "Operador"])
            
            # Escribir datos
            for channel in channels:
                writer.writerow([
                    channel["frequency"]/1e6,
                    10*np.log10(channel["power"]),
                    channel.get("bandwidth", 0),
                    channel["timestamp"],
                    channel.get("classification", "Unknown"),
                    channel.get("confidence", 0.0),
                    channel.get("mcc", ""),
                    channel.get("mnc", ""),
                    self._get_operator_name(channel.get("operator_id"))
                ])
        
        self.logger.info(f"Resultados guardados en {filename} y {csv_filename}")
        print(f"Resultados guardados en {filename} y {csv_filename}")
    
    def _get_operator_name(self, operator_id):
        """Obtiene el nombre del operador a partir de su ID"""
        if operator_id is None:
            return "Desconocido"
        
        for operator in self.operators_db["operators"]:
            if operator["id"] == operator_id:
                return f"{operator['name']} ({operator['country']})"
        
        return "Desconocido"
    
    def _generate_scan_report(self, channels):
        """Genera un informe HTML con los resultados del escaneo"""
        if not channels:
            return
        
        # Crear directorio de informes si no existe
        if not os.path.exists(self.reports_directory):
            os.makedirs(self.reports_directory)
        
        # Nombre del archivo de informe
        report_file = os.path.join(self.reports_directory, f"scan_report_{time.strftime('%Y%m%d_%H%M%S')}.html")
        
        # Generar gráficos para el informe
        # 1. Gráfico de frecuencias vs potencia
        plt.figure(figsize=(10, 6))
        freqs = [c["frequency"]/1e6 for c in channels]
        powers = [10 * np.log10(c["power"]) for c in channels]
        plt.bar(freqs, powers)
        plt.xlabel('Frecuencia (MHz)')
        plt.ylabel('Potencia (dB)')
        plt.title('Señales detectadas por frecuencia')
        plt.grid(True)
        
        freq_chart_file = f"freq_chart_{time.strftime('%Y%m%d_%H%M%S')}.png"
        freq_chart_path = os.path.join(self.reports_directory, freq_chart_file)
        plt.savefig(freq_chart_path)
        plt.close()
        
        # 2. Gráfico de distribución por tipo de señal
        signal_types = {}
        for channel in channels:
            sig_type = channel.get("classification", "Unknown")
            if sig_type in signal_types:
                signal_types[sig_type] += 1
            else:
                signal_types[sig_type] = 1
        
        plt.figure(figsize=(8, 8))
        plt.pie(signal_types.values(), labels=signal_types.keys(), autopct='%1.1f%%')
        plt.title('Distribución por tipo de señal')
        
        types_chart_file = f"types_chart_{time.strftime('%Y%m%d_%H%M%S')}.png"
        types_chart_path = os.path.join(self.reports_directory, types_chart_file)
        plt.savefig(types_chart_path)
        plt.close()
        
        # Generar HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Informe de Escaneo TETRA - {time.strftime('%Y-%m-%d %H:%M:%S')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .chart img {{ max-width: 100%; height: auto; }}
                .footer {{ margin-top: 30px; font-size: 0.8em; color: #7f8c8d; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Informe de Escaneo TETRA</h1>
            <p><strong>Fecha y hora:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Dispositivo:</strong> {self.device}</p>
            <p><strong>Rango de frecuencias:</strong> {min(freqs):.3f} - {max(freqs):.3f} MHz</p>
            <p><strong>Señales detectadas:</strong> {len(channels)}</p>
            
            <h2>Distribución de Señales</h2>
            <div class="chart">
                <img src="{freq_chart_file}" alt="Gráfico de frecuencias">
            </div>
            
            <div class="chart">
                <img src="{types_chart_file}" alt="Distribución por tipo">
            </div>
            
            <h2>Detalles de Señales Detectadas</h2>
            <table>
                <tr>
                    <th>Frecuencia (MHz)</th>
                    <th>Potencia (dB)</th>
                    <th>Ancho de banda (kHz)</th>
                    <th>Tipo</th>
                    <th>Confianza</th>
                    <th>MCC</th>
                    <th>MNC</th>
                    <th>Operador</th>
                </tr>
        """
        
        # Añadir filas de la tabla
        for channel in channels:
            html_content += f"""
                <tr>
                    <td>{channel["frequency"]/1e6:.6f}</td>
                    <td>{10*np.log10(channel["power"]):.2f}</td>
                    <td>{channel.get("bandwidth", 0):.1f}</td>
                    <td>{channel.get("classification", "Unknown")}</td>
                    <td>{channel.get("confidence", 0.0):.2f}</td>
                    <td>{channel.get("mcc", "")}</td>
                    <td>{channel.get("mnc", "")}</td>
                    <td>{self._get_operator_name(channel.get("operator_id"))}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="footer">
                <p>Generado por TETRA Monitor Pro - Solo para uso educativo y legal</p>
            </div>
        </body>
        </html>
        """
        
        # Guardar HTML
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Informe generado: {report_file}")
        print(f"Informe generado: {report_file}")
        
        # Abrir informe en navegador
        webbrowser.open('file://' + os.path.abspath(report_file))
    
    def monitor_frequency(self, duration=60, record_audio=False, advanced_mode=False):
        """Monitorea una frecuencia específica y muestra análisis espectral"""
        self.logger.info(f"Monitoreando {self.freq/1e6} MHz durante {duration} segundos...")
        print(f"Monitoreando {self.freq/1e6} MHz durante {duration} segundos...")
        
        # Configurar gráficos
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        plt.ion()
        
        # Configurar grabación de audio si está habilitada
        if record_audio:
            self._setup_audio_recording()
        
        start_time = time.time()
        self.running = True
        
        # Preparar para visualización de espectrograma
        spectrogram_data = []
        
        # Obtener ubicación GPS si está disponible
        location_info = self._get_gps_location()
        
        # Crear hash único para esta sesión de monitoreo
        session_hash = hashlib.md5(f"{self.freq}_{time.time()}".encode()).hexdigest()
        
        # Identificar operador
        operator_id, mcc, mnc = self._identify_operator(self.freq)
        
        # Guardar señal en base de datos
        signal_id = self._save_signal_to_db({
            "frequency": self.freq,
            "power": 0.0,  # Se actualizará durante el monitoreo
            "bandwidth": 0.0,  # Se actualizará durante el monitoreo
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "location": json.dumps(location_info) if location_info else None,
            "classification": "Monitoring",
            "hash": session_hash,
            "operator_id": operator_id,
            "mcc": mcc,
            "mnc": mnc
        })
        
        # Guardar localización si está disponible
        if location_info:
            self._save_location_to_db(signal_id, location_info)
        
        # Variables para análisis de señal
        constellation_points = []
        symbol_rate_estimate = 0
        
        try:
            while self.running and (time.time() - start_time < duration):
                # Obtener muestras
                samples = self.sdr.get_samples(8192)
                
                # Análisis espectral
                spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2
                freq_range = np.linspace(self.freq - self.sample_rate/2, 
                                        self.freq + self.sample_rate/2, 
                                        len(spectrum)) / 1e6
                
                # Guardar datos para espectrograma
                spectrogram_data.append(10 * np.log10(spectrum))
                
                # Calcular potencia actual
                current_power = np.mean(np.abs(samples)**2)
                
                # Estimar ancho de banda
                bandwidth = self._estimate_bandwidth(spectrum, freq_range)
                
                # Actualizar señal en base de datos periódicamente
                if int(time.time() - start_time) % 10 == 0:
                    cursor = self.conn.cursor()
                    cursor.execute('''
                    UPDATE signals SET power = ?, bandwidth = ? WHERE id = ?
                    ''', (current_power, bandwidth, signal_id))
                    self.conn.commit()
                
                # Visualizar espectro
                ax1.clear()
                ax1.plot(freq_range, 10 * np.log10(spectrum))
                ax1.grid(True)
                ax1.set_xlabel('Frecuencia (MHz)')
                ax1.set_ylabel('Potencia (dB)')
                ax1.set_title(f'Análisis Espectral - {self.freq/1e6} MHz - Potencia: {10*np.log10(current_power):.2f} dB')
                
                # Visualizar espectrograma
                if len(spectrogram_data) > 10:  # Esperar a tener suficientes datos
                    ax2.clear()
                    ax2.imshow(np.array(spectrogram_data)[-100:], 
                              aspect='auto', 
                              extent=[freq_range[0], freq_range[-1], 0, 10],
                              origin='lower',
                              cmap='viridis')
                    ax2.set_xlabel('Frecuencia (MHz)')
                    ax2.set_ylabel('Tiempo (s)')
                    ax2.set_title('Espectrograma')
                
                # Visualizar constelación (solo en modo avanzado)
                if advanced_mode:
                    # Calcular puntos de constelación
                    if len(constellation_points) < 1000:
                        # Submuestrear para reducir número de puntos
                        constellation_points.extend(samples[::20])
                    else:
                        constellation_points = constellation_points[-1000:]
                        constellation_points.extend(samples[::20])
                        constellation_points = constellation_points[-1000:]
                    
                    # Visualizar
                    ax3.clear()
                    ax3.scatter(np.real(constellation_points), np.imag(constellation_points), 
                               s=2, alpha=0.5, c='blue')
                    ax3.set_xlim(-2, 2)
                    ax3.set_ylim(-2, 2)
                    ax3.grid(True)
                    ax3.set_xlabel('I')
                    ax3.set_ylabel('Q')
                    ax3.set_title('Diagrama de Constelación')
                    
                    # Análisis de protocolo periódico
                    if int(time.time() - start_time) % 5 == 0:
                        self._analyze_protocol(samples, self.freq, signal_id)
                
                plt.tight_layout()
                plt.pause(0.1)
                
                # Guardar captura cada 5 segundos
                if int(time.time() - start_time) % 5 == 0:
                    self._save_spectrum_data(freq_range, spectrum)
                    
                    # Análisis de actividad
                    signal_power = np.mean(np.abs(samples)**2)
                    if signal_power > 0.01:  # Umbral de actividad
                        self.logger.info(f"Actividad detectada en {self.freq/1e6:.3f} MHz (potencia: {signal_power:.6f})")
                        
                        # Análisis de patrón TETRA (4 slots por trama)
                        self._analyze_tetra_pattern(samples)
        
        except KeyboardInterrupt:
            self.logger.info("Monitoreo interrumpido por el usuario")
            print("Monitoreo interrumpido por el usuario")
        finally:
            self.running = False
            if record_audio:
                self._stop_audio_recording()
            plt.ioff()
            plt.close()
            
            # Guardar espectrograma completo
            if spectrogram_data:
                self._save_spectrogram(spectrogram_data, freq_range)
    
    def _analyze_tetra_pattern(self, samples):
        """Analiza patrones típicos de TETRA en las muestras"""
        # Implementación básica para detectar estructura de trama TETRA
        # Esto es una simplificación educativa
        
        # Demodulación básica (solo para análisis de patrón, no decodificación)
        demod = np.abs(samples[1:] * np.conj(samples[:-1]))
        
        # Buscar periodicidad característica de TETRA (4 slots por trama)
        autocorr = np.correlate(demod, demod, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Análisis básico de periodicidad
        peaks = signal.find_peaks(autocorr)[0]
        if len(peaks) > 3:
            intervals = np.diff(peaks[:4])
            avg_interval = np.mean(intervals)
            
            # Verificar si el patrón es consistente con TETRA
            if 0.9 < np.std(intervals)/avg_interval < 1.1:
                self.logger.info(f"Patrón consistente con estructura TETRA detectado")
                print(f"Patrón consistente con estructura TETRA detectado")
    
    def _setup_audio_recording(self):
        """Configura la grabación de audio demodulado"""
        self.audio_filename = os.path.join(
            self.audio_directory,
            f"tetra_{self.freq/1e6:.3f}MHz_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        )
        self.recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        self.logger.info(f"Grabación de audio iniciada: {self.audio_filename}")
        print(f"Grabación de audio iniciada: {self.audio_filename}")
    
    def _record_audio(self):
        """Función para grabar audio demodulado en un hilo separado"""
        # Implementación básica - en un sistema real requeriría demodulación específica para TETRA
        # Esta es una versión simplificada para fines educativos
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=48000,
                        output=True)
        
        # Abrir archivo WAV para grabación
        wavfile = wave.open(self.audio_filename, 'wb')
        wavfile.setnchannels(1)
        wavfile.setsampwidth(4)  # 32 bits = 4 bytes
        wavfile.setframerate(48000)
        
        while self.recording:
            # Obtener muestras
            samples = self.sdr.get_samples(1024)
            
            # Demodulación FM básica (simplificada)
            demod = np.angle(samples[1:] * np.conj(samples[:-1]))
            
            # Normalizar y reproducir
            audio = demod / np.max(np.abs(demod)) if np.max(np.abs(demod)) > 0 else demod
            audio = audio * 0.5
            
            # Reproducir audio
            stream.write(audio.astype(np.float32).tobytes())
            
            # Guardar en archivo WAV
            wavfile.writeframes((audio * 32767).astype(np.int16).tobytes())
            
            time.sleep(0.01)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        wavfile.close()
    
    def _stop_audio_recording(self):
        """Detiene la grabación de audio"""
        self.recording = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0)
        self.logger.info("Grabación de audio detenida")
        print("Grabación de audio detenida")
    
    def _save_spectrum_data(self, freqs, spectrum):
        """Guarda datos del espectro para análisis posterior"""
        filename = os.path.join(self.spectrum_directory, 
                              f"spectrum_{self.freq/1e6:.3f}MHz_{time.strftime('%H%M%S')}.npz")
        np.savez(filename, frequencies=freqs, power=spectrum)
    
    def _save_spectrogram(self, spectrogram_data, freq_range):
        """Guarda el espectrograma completo como imagen y datos"""
        # Guardar imagen
        plt.figure(figsize=(10, 6))
        plt.imshow(np.array(spectrogram_data), 
                  aspect='auto', 
                  extent=[freq_range[0], freq_range[-1], 0, len(spectrogram_data)],
                  origin='lower',
                  cmap='viridis')
        plt.colorbar(label='Potencia (dB)')
        plt.xlabel('Frecuencia (MHz)')
        plt.ylabel('Tiempo (muestras)')
        plt.title(f'Espectrograma TETRA - {self.freq/1e6:.3f} MHz')
        
        img_filename = os.path.join(self.spectrum_directory, 
                                  f"spectrogram_{self.freq/1e6:.3f}MHz_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(img_filename, dpi=300)
        plt.close()
        
        # Guardar datos
        data_filename = os.path.join(self.spectrum_directory, 
                                  f"spectrogram_data_{self.freq/1e6:.3f}MHz_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        np.savez(data_filename, spectrogram=np.array(spectrogram_data), frequencies=freq_range)
        
        self.logger.info(f"Espectrograma guardado: {img_filename}")
        print(f"Espectrograma guardado: {img_filename}")
    
    def start_web_server(self, port=8080):
        """Inicia un servidor web para visualización remota"""
        if self.web_server_running:
            print("El servidor web ya está en ejecución")
            return
        
        # Definir manejador de peticiones HTTP
        class TetraMonitorHandler(BaseHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.tetra_monitor = kwargs.pop('tetra_monitor')
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    # Generar página principal
                    html = self._generate_main_page()
                    self.wfile.write(html.encode())
                
                elif self.path == '/data':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    # Obtener datos más recientes
                    data = self._get_recent_data()
                    self.wfile.write(json.dumps(data).encode())
                
                elif self.path.startswith('/spectrum/'):
                    try:
                        # Extraer ID de la señal
                        signal_id = int(self.path.split('/')[-1])
                        
                        # Obtener datos del espectro
                        spectrum_data = self._get_spectrum_data(signal_id)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'image/png')
                        self.end_headers()
                        
                        # Generar imagen del espectro
                        img_data = self._generate_spectrum_image(spectrum_data)
                        self.wfile.write(img_data)
                    except:
                        self.send_response(404)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def _generate_main_page(self):
                """Genera la página HTML principal"""
                cursor = self.tetra_monitor.conn.cursor()
                
                # Obtener señales recientes
                cursor.execute('''
                SELECT id, frequency, power, classification, timestamp 
                FROM signals ORDER BY id DESC LIMIT 20
                ''')
                
                signals = cursor.fetchall()
                
                # Generar HTML
                html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>TETRA Monitor Pro - Interfaz Web</title>
                    <meta http-equiv="refresh" content="30">
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #2c3e50; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .footer {{ margin-top: 30px; font-size: 0.8em; color: #7f8c8d; text-align: center; }}
                    </style>
                </head>
                <body>
                    <h1>TETRA Monitor Pro - Interfaz Web</h1>
                    <p>Última actualización: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <h2>Señales Recientes</h2>
                    <table>
                        <tr>
                            <th>ID</th>
                            <th>Frecuencia (MHz)</th>
                            <th>Potencia (dB)</th>
                            <th>Tipo</th>
                            <th>Timestamp</th>
                            <th>Acciones</th>
                        </tr>
                '''
                
                for signal in signals:
                    html += f'''
                        <tr>
                            <td>{signal[0]}</td>
                            <td>{signal[1]/1e6:.6f}</td>
                            <td>{10*np.log10(signal[2]) if signal[2] > 0 else "N/A"}</td>
                            <td>{signal[3]}</td>
                            <td>{signal[4]}</td>
                            <td><a href="/spectrum/{signal[0]}" target="_blank">Ver Espectro</a></td>
                        </tr>
                    '''
                
                html += '''
                    </table>
                    
                    <div class="footer">
                        <p>TETRA Monitor Pro - Solo para uso educativo y legal</p>
                    </div>
                </body>
                </html>
                '''
                
                return html
            
            def _get_recent_data(self):
                """Obtiene datos recientes para API JSON"""
                cursor = self.tetra_monitor.conn.cursor()
                
                # Obtener señales recientes
                cursor.execute('''
                SELECT id, frequency, power, bandwidth, classification, timestamp 
                FROM signals ORDER BY id DESC LIMIT 50
                ''')
                
                signals = []
                for row in cursor.fetchall():
                    signals.append({
                        "id": row[0],
                        "frequency": row[1]/1e6,
                        "power": 10*np.log10(row[2]) if row[2] > 0 else -100,
                        "bandwidth": row[3],
                        "type": row[4],
                        "timestamp": row[5]
                    })
                
                # Obtener alertas recientes
                cursor.execute('''
                SELECT id, type, description, frequency, timestamp 
                FROM alerts WHERE acknowledged = 0 ORDER BY id DESC LIMIT 10
                ''')
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        "id": row[0],
                        "type": row[1],
                        "description": row[2],
                        "frequency": row[3]/1e6 if row[3] else 0,
                        "timestamp": row[4]
                    })
                
                return {
                    "signals": signals,
                    "alerts": alerts,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            def _get_spectrum_data(self, signal_id):
                """Obtiene datos del espectro para una señal específica"""
                cursor = self.tetra_monitor.conn.cursor()
                
                # Obtener detalles de la señal
                cursor.execute('''
                SELECT frequency, power, bandwidth, classification 
                FROM signals WHERE id = ?
                ''', (signal_id,))
                
                signal = cursor.fetchone()
                if not signal:
                    return None
                
                # Obtener datos de protocolo si existen
                cursor.execute('''
                SELECT raw_data FROM protocol_data 
                WHERE signal_id = ? ORDER BY id DESC LIMIT 1
                ''', (signal_id,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                # Convertir datos binarios a numpy array
                raw_data = np.frombuffer(result[0], dtype=np.complex64)
                
                return {
                    "signal": signal,
                    "raw_data": raw_data
                }
            
            def _generate_spectrum_image(self, spectrum_data):
                """Genera una imagen del espectro"""
                if not spectrum_data:
                    # Generar imagen de error
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.text(0.5, 0.5, "Datos no disponibles", ha='center', va='center', fontsize=14)
                    ax.axis('off')
                    
                    img_data = BytesIO()
                    fig.savefig(img_data, format='png')
                    plt.close(fig)
                    img_data.seek(0)
                    return img_data.read()
                
                # Extraer datos
                signal = spectrum_data["signal"]
                raw_data = spectrum_data["raw_data"]
                
                # Calcular espectro
                spectrum = np.abs(np.fft.fftshift(np.fft.fft(raw_data)))**2
                freq = signal[0]
                sample_rate = 2e6  # Valor predeterminado
                freq_range = np.linspace(freq - sample_rate/2, freq + sample_rate/2, len(spectrum)) / 1e6
                
                # Crear figura
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Gráfico de espectro
                ax1.plot(freq_range, 10 * np.log10(spectrum))
                ax1.grid(True)
                ax1.set_xlabel('Frecuencia (MHz)')
                ax1.set_ylabel('Potencia (dB)')
                ax1.set_title(f'Espectro - {freq/1e6:.3f} MHz - {signal[3]}')
                
                # Constelación
                ax2.scatter(np.real(raw_data[::10]), np.imag(raw_data[::10]), s=2, alpha=0.5)
                ax2.set_xlim(-2, 2)
                ax2.set_ylim(-2, 2)
                ax2.grid(True)
                ax2.set_xlabel('I')
                ax2.set_ylabel('Q')
                ax2.set_title('Diagrama de Constelación')
                
                plt.tight_layout()
                
                # Guardar imagen en memoria
                img_data = BytesIO()
                fig.savefig(img_data, format='png', dpi=100)
                plt.close(fig)
                img_data.seek(0)
                
                return img_data.read()
        
        # Crear servidor en un hilo separado
        def run_server():
            handler = lambda *args, **kwargs: TetraMonitorHandler(*args, tetra_monitor=self, **kwargs)
            server = HTTPServer(('', port), handler)
            self.logger.info(f"Servidor web iniciado en http://localhost:{port}")
            print(f"Servidor web iniciado en http://localhost:{port}")
            server.serve_forever()
        
        self.web_server_thread = threading.Thread(target=run_server)
        self.web_server_thread.daemon = True
        self.web_server_thread.start()
        self.web_server_running = True
    
    def stop_web_server(self):
        """Detiene el servidor web"""
        if not self.web_server_running:
            return
        
        # No hay forma directa de detener el servidor HTTP, 
        # pero como el hilo es daemon, terminará cuando el programa principal termine
        self.web_server_running = False
        self.logger.info("Servidor web detenido")
        print("Servidor web detenido")
    
    def analyze_database(self):
        """Analiza los datos recopilados en la base de datos"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT frequency, COUNT(*) as count, AVG(power) as avg_power FROM signals GROUP BY frequency ORDER BY count DESC")
        results = cursor.fetchall()
        
        if not results:
            print("No hay datos suficientes en la base de datos para análisis")
            return
        
        print("\nAnálisis de frecuencias más activas:")
        print("------------------------------------")
        print("Frecuencia (MHz) | Detecciones | Potencia media (dB)")
        print("------------------------------------")
        
        for freq, count, power in results[:10]:  # Top 10
            print(f"{freq/1e6:14.3f} | {count:11d} | {10*np.log10(power):17.2f}")
        
        # Visualizar resultados
        plt.figure(figsize=(10, 6))
        freqs = [r[0]/1e6 for r in results]
        counts = [r[1] for r in results]
        plt.bar(freqs, counts)
        plt.xlabel('Frecuencia (MHz)')
        plt.ylabel('Número de detecciones')
        plt.title('Actividad por frecuencia')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.log_directory, f"frequency_activity_{time.strftime('%Y%m%d')}.png"))
        plt.show()
        
        # Análisis por tipo de señal
        cursor.execute("SELECT classification, COUNT(*) as count FROM signals GROUP BY classification ORDER BY count DESC")
        type_results = cursor.fetchall()
        
        if type_results:
            print("\nAnálisis por tipo de señal:")
            print("-------------------------")
            print("Tipo           | Detecciones")
            print("-------------------------")
            
            for type_name, count in type_results:
                print(f"{type_name:14s} | {count:11d}")
            
            # Visualizar resultados
            plt.figure(figsize=(8, 8))
            types = [r[0] for r in type_results]
            counts = [r[1] for r in type_results]
            plt.pie(counts, labels=types, autopct='%1.1f%%')
            plt.title('Distribución por tipo de señal')
            
            plt.savefig(os.path.join(self.log_directory, f"signal_types_{time.strftime('%Y%m%d')}.png"))
            plt.show()
    
    def export_database(self, format="csv"):
        """Exporta la base de datos a CSV o JSON"""
        if format not in ["csv", "json"]:
            print("Formato no soportado. Use 'csv' o 'json'")
            return
        
        # Crear directorio de exportación si no existe
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)
        
        # Exportar tabla de señales
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT id, frequency, power, bandwidth, timestamp, location, classification, confidence, 
               hash, mcc, mnc, operator_id
        FROM signals
        ''')
        
        signals = cursor.fetchall()
        
        if format == "csv":
            filename = os.path.join(self.export_directory, f"signals_export_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Escribir encabezados
                writer.writerow(["ID", "Frecuencia (MHz)", "Potencia (dB)", "Ancho de banda (kHz)", 
                               "Timestamp", "Ubicación", "Clasificación", "Confianza", 
                               "Hash", "MCC", "MNC", "Operador"])
                
                # Escribir datos
                for signal in signals:
                    writer.writerow([
                        signal[0],
                        signal[1]/1e6,
                        10*np.log10(signal[2]) if signal[2] > 0 else -100,
                        signal[3],
                        signal[4],
                        signal[5],
                        signal[6],
                        signal[7],
                        signal[8],
                        signal[9],
                        signal[10],
                        self._get_operator_name(signal[11])
                    ])
        else:  # JSON
            filename = os.path.join(self.export_directory, f"signals_export_{time.strftime('%Y%m%d_%H%M%S')}.json")
            data = []
            for signal in signals:
                data.append({
                    "id": signal[0],
                    "frequency": signal[1]/1e6,
                    "power_db": 10*np.log10(signal[2]) if signal[2] > 0 else -100,
                    "bandwidth": signal[3],
                    "timestamp": signal[4],
                    "location": signal[5],
                    "classification": signal[6],
                    "confidence": signal[7],
                    "hash": signal[8],
                    "mcc": signal[9],
                    "mnc": signal[10],
                    "operator": self._get_operator_name(signal[11])
                })
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        self.logger.info(f"Base de datos exportada a {filename}")
        print(f"Base de datos exportada a {filename}")
    
    def close(self):
        """Cierra los recursos del receptor y la base de datos"""
        self.running = False
        self.recording = False
        
        # Detener servidor web si está en ejecución
        if self.web_server_running:
            self.stop_web_server()
        
        # Cerrar sesión
        self._close_session()
        
        if hasattr(self, 'sdr'):
            self.sdr.close()
        
        if hasattr(self, 'conn'):
            self.conn.close()
            
        self.logger.info("Recursos liberados y aplicación cerrada")
        print("Recursos liberados y aplicación cerrada")

def main():
    parser = argparse.ArgumentParser(description='TETRA Monitor Pro - Herramienta avanzada para análisis de señales TETRA')
    parser.add_argument('-f', '--freq', type=float, default=395.0,
                        help='Frecuencia central en MHz (default: 395.0)')
    parser.add_argument('-s', '--scan', action='store_true',
                        help='Escanear banda TETRA completa')
    parser.add_argument('--start', type=float, default=380.0,
                        help='Frecuencia inicial para escaneo en MHz (default: 380.0)')
    parser.add_argument('--end', type=float, default=400.0,
                        help='Frecuencia final para escaneo en MHz (default: 400.0)')
    parser.add_argument('--step', type=float, default=0.025,
                        help='Paso de frecuencia para escaneo en MHz (default: 0.025)')
    parser.add_argument('-d', '--device', type=str, default='airspy',
                        help='Dispositivo SDR (airspy, hackrf, rtlsdr)')
    parser.add_argument('-g', '--gain', type=int, default=30,
                        help='Ganancia del receptor (default: 30)')
    parser.add_argument('-t', '--time', type=int, default=60,
                        help='Duración del monitoreo en segundos (default: 60)')
    parser.add_argument('-a', '--audio', action='store_true',
                        help='Grabar audio demodulado')
    parser.add_argument('--advanced', action='store_true',
                        help='Activar modo avanzado de análisis')
    parser.add_argument('--analyze', action='store_true',
                        help='Analizar datos recopilados en la base de datos')
    parser.add_argument('--export', choices=['csv', 'json'],
                        help='Exportar base de datos a CSV o JSON')
    parser.add_argument('--web', action='store_true',
                        help='Iniciar servidor web para visualización remota')
    parser.add_argument('--port', type=int, default=8080,
                        help='Puerto para servidor web (default: 8080)')
    
    args = parser.parse_args()
    
    monitor = TetraMonitorPro(
        freq=args.freq * 1e6,
        gain=args.gain,
        device=args.device
    )
    
    try:
        if args.web:
            monitor.start_web_server(port=args.port)
        
        if args.analyze:
            monitor.analyze_database()
        elif args.export:
            monitor.export_database(format=args.export)
        else:
            if not monitor.setup_receiver():
                return
                
            if args.scan:
                monitor.scan_tetra_band(
                    start_freq=args.start * 1e6,
                    end_freq=args.end * 1e6,
                    step=args.step * 1e6,
                    advanced_mode=args.advanced
                )
            else:
                monitor.monitor_frequency(
                    duration=args.time,
                    record_audio=args.audio,
                    advanced_mode=args.advanced
                )
    finally:
        monitor.close()

if __name__ == "__main__":
    print("TETRA Monitor Pro - SOLO PARA USO EDUCATIVO Y LEGAL")
    print("Esta herramienta está diseñada para monitoreo pasivo y análisis")
    print("El uso debe cumplir con las regulaciones locales sobre telecomunicaciones")
    main()
