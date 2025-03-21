import sys
import os

# matplotlib backend 설정을 가장 먼저 수행
os.environ['MPLBACKEND'] = 'Qt5Agg'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 기본 라이브러리 import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# TensorFlow import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# PyQt5 import
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                           QTextEdit, QFileDialog, QMessageBox, QProgressDialog,
                           QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# matplotlib import를 마지막에 배치
import matplotlib
matplotlib.use('Qt5Agg')  # backend 설정을 명시적으로 수행
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# GPU 메모리 증가 방지
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 메모리 설정 중 오류 발생: {e}")

def simulate_combat(hp1, attack1, defense1, hp2, attack2, defense2):
    """
    전투를 시뮬레이션하여 전투 시간과 승자를 반환합니다.
    데미지 계산식: 실제 데미지 = max(1, 공격력 - 방어력)
    """   
    current_hp1 = hp1
    current_hp2 = hp2
    time = 0
    
    # 실제 데미지 계산
    damage1 = max(1, attack1 - defense2)
    damage2 = max(1, attack2 - defense1)
    
    while current_hp1 > 0 and current_hp2 > 0:
        # 양쪽이 동시에 공격
        current_hp1 -= damage2
        current_hp2 -= damage1
        time += 1
    
    winner = "PC1" if current_hp2 <= 0 else "PC2"
    if current_hp1 <= 0 and current_hp2 <= 0:
        winner = "Draw"
    
    return time, winner

class TrainingThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    loss_update = pyqtSignal(dict)
    
    def __init__(self, combat_data, epochs=100):
        super().__init__()
        self.combat_data = combat_data
        self.epochs = epochs
        self.model = None
        
    def run(self):
        try:
            # 메모리 설정
            tf.keras.backend.clear_session()
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    self.error.emit(f"GPU 설정 오류: {str(e)}")
                    return

            if self.combat_data is None or len(self.combat_data) == 0:
                self.error.emit("데이터가 비어있습니다.")
                return
            
            try:
                # 전투 시뮬레이션 실행
                combat_times = []
                winners = []
                total_rows = len(self.combat_data)
                
                for idx, row in self.combat_data.iterrows():
                    try:
                        time, winner = simulate_combat(
                            float(row['HP1']), float(row['Attack1']), float(row['Defense1']),
                            float(row['HP2']), float(row['Attack2']), float(row['Defense2'])
                        )
                        combat_times.append(time)
                        winners.append(winner)
                        
                        if idx % 10 == 0:
                            progress = int((idx + 1) / total_rows * 50)
                            self.progress.emit(progress)
                    except Exception as e:
                        self.error.emit(f"전투 시뮬레이션 오류 (행 {idx}): {str(e)}")
                        return
                
                # 결과 데이터프레임에 추가
                self.combat_data['Total_Time'] = combat_times
                self.combat_data['Winner'] = winners
                
                # 데이터 준비
                X = self.combat_data[['HP1', 'Attack1', 'Defense1', 'HP2', 'Attack2', 'Defense2']].astype('float32')
                y_time = self.combat_data['Total_Time'].astype('float32')
                # 승자 예측을 위한 이진 레이블 생성 (PC1이 이기면 1, PC2가 이기면 0)
                y_winner = (self.combat_data['Winner'] == 'PC1').astype('float32')
                
                # 데이터 정규화
                X_mean = X.mean()
                X_std = X.std()
                X_normalized = (X - X_mean) / X_std
                
                # 데이터 분할
                X_train, X_test, y_time_train, y_time_test, y_winner_train, y_winner_test = train_test_split(
                    X_normalized, y_time, y_winner, test_size=0.2, random_state=42
                )
                
                # 멀티태스크 모델 생성 (전투 시간과 승자 예측)
                inputs = layers.Input(shape=(6,))
                shared = layers.Dense(64, activation='relu')(inputs)
                shared = layers.Dense(32, activation='relu')(shared)
                
                # 전투 시간 예측 브랜치
                time_branch = layers.Dense(16, activation='relu')(shared)
                time_output = layers.Dense(1, name='time_output')(time_branch)
                
                # 승자 예측 브랜치 (이진 분류)
                winner_branch = layers.Dense(16, activation='relu')(shared)
                winner_output = layers.Dense(1, activation='sigmoid', name='winner_output')(winner_branch)
                
                self.model = keras.Model(inputs=inputs, outputs=[time_output, winner_output])
                
                # 모델 컴파일
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss={
                        'time_output': 'mse',
                        'winner_output': 'binary_crossentropy'
                    },
                    metrics={
                        'time_output': ['mae'],
                        'winner_output': ['accuracy']
                    }
                )
                
                # 콜백 설정
                class CustomCallback(keras.callbacks.Callback):
                    def __init__(self, progress_signal, loss_signal, total_epochs):
                        super().__init__()
                        self.progress_signal = progress_signal
                        self.loss_signal = loss_signal
                        self.total_epochs = total_epochs
                        self.history = {
                            'time_loss': [],
                            'winner_loss': [],
                            'time_val_loss': [],
                            'winner_val_loss': [],
                            'time_mae': [],
                            'winner_accuracy': [],
                            'time_val_mae': [],
                            'winner_val_accuracy': []
                        }
                    
                    def on_epoch_end(self, epoch, logs=None):
                        try:
                            # 진행률 업데이트
                            progress = 50 + int((epoch + 1) / self.total_epochs * 50)
                            self.progress_signal.emit(progress)
                            
                            # logs가 None인 경우 빈 딕셔너리로 초기화
                            logs = logs or {}
                            
                            # 현재 에포크의 메트릭 값들을 저장
                            metrics = {
                                'time_loss': logs.get('time_output_loss', 0.0),
                                'winner_loss': logs.get('winner_output_loss', 0.0),
                                'time_val_loss': logs.get('val_time_output_loss', 0.0),
                                'winner_val_loss': logs.get('val_winner_output_loss', 0.0),
                                'time_mae': logs.get('time_output_mae', 0.0),
                                'winner_accuracy': logs.get('winner_output_accuracy', 0.0),
                                'time_val_mae': logs.get('val_time_output_mae', 0.0),
                                'winner_val_accuracy': logs.get('val_winner_output_accuracy', 0.0)
                            }
                            
                            # 모든 메트릭을 history에 추가
                            for key, value in metrics.items():
                                self.history[key].append(value)
                            
                            # 모든 리스트의 길이가 같은지 확인하고 동기화
                            max_length = max(len(values) for values in self.history.values())
                            
                            # 길이가 다른 리스트들을 마지막 값으로 채움
                            for key in self.history:
                                current_length = len(self.history[key])
                                if current_length < max_length:
                                    last_value = self.history[key][-1] if self.history[key] else 0.0
                                    self.history[key].extend([last_value] * (max_length - current_length))
                            
                            # 모든 리스트가 같은 길이를 가지고 있는지 다시 확인
                            lengths = [len(values) for values in self.history.values()]
                            if len(set(lengths)) != 1:
                                print(f"Warning: History lengths still not synchronized: {lengths}")
                            
                            # 손실값 업데이트 신호 발생
                            self.loss_signal.emit({
                                'current_epoch': epoch + 1,
                                **{k: v.copy() for k, v in self.history.items()}  # 깊은 복사를 통해 데이터 전달
                            })
                            
                        except Exception as e:
                            print(f"Error in CustomCallback.on_epoch_end: {str(e)}")
                            # 에러가 발생해도 학습은 계속 진행되도록 함
                            pass
                
                callbacks = [
                    CustomCallback(self.progress, self.loss_update, self.epochs),
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
                
                # 학습 실행
                history = self.model.fit(
                    X_train, 
                    {
                        'time_output': y_time_train,
                        'winner_output': y_winner_train
                    },
                    validation_data=(
                        X_test,
                        {
                            'time_output': y_time_test,
                            'winner_output': y_winner_test
                        }
                    ),
                    epochs=self.epochs,
                    batch_size=32,
                    verbose=0,
                    callbacks=callbacks
                )
                
                # 결과 반환
                result = {
                    'model': self.model,
                    'history': history,
                    'combat_data': self.combat_data,
                    'X_mean': X_mean,
                    'X_std': X_std
                }
                self.finished.emit(result)

            except Exception as e:
                self.error.emit(f"모델 학습 오류: {str(e)}")
                return
                
        except Exception as e:
            self.error.emit(f"처리 중 오류 발생: {str(e)}")
            return
        finally:
            # 메모리 정리
            try:
                tf.keras.backend.clear_session()
            except:
                pass
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.model:
                del self.model
            tf.keras.backend.clear_session()
        except:
            pass

class OptimizationThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, model, target_time, steps=100):
        super().__init__()
        self.model = model
        self.target_time = target_time
        self.steps = steps
    
    def run(self):
        try:
            # 텐서플로우 연산을 위한 변수 설정
            with tf.device('/CPU:0'):  # CPU에서 실행
                attack = tf.Variable(50.0, dtype=tf.float32)
                defense = tf.Variable(20.0, dtype=tf.float32)
                optimizer = tf.optimizers.Adam(learning_rate=0.1)
                
                optimization_results = []
                
                for i in range(self.steps):
                    with tf.GradientTape() as tape:
                        input_data = tf.stack([
                            tf.constant(1000.0),
                            attack,
                            defense,
                            tf.constant(900.0),
                            tf.constant(55.0),
                            tf.constant(18.0)
                        ])
                        input_data = tf.reshape(input_data, [1, 6])
                        pred_time = self.model(input_data, training=False)[0][0]
                        loss = tf.abs(pred_time - self.target_time)
                    
                    gradients = tape.gradient(loss, [attack, defense])
                    if gradients[0] is not None and gradients[1] is not None:
                        optimizer.apply_gradients(zip(gradients, [attack, defense]))
                    
                    if i % 10 == 0:
                        optimization_results.append(
                            f"Step {i}: Attack = {attack.numpy():.2f}, "
                            f"Defense = {defense.numpy():.2f}, "
                            f"Predicted Time = {pred_time.numpy():.2f}\n"
                        )
                    
                    # 진행률 업데이트
                    progress = int((i + 1) / self.steps * 100)
                    self.progress.emit(progress)
                
                result = {
                    'attack': float(attack.numpy()),
                    'defense': float(defense.numpy()),
                    'results': optimization_results
                }
                self.finished.emit(result)
                
        except Exception as e:
            self.error.emit(f"최적화 중 오류 발생: {str(e)}")
            return

def analyze_balance_stats(combat_data):
    """
    Analyze combat data balance and return statistics
    """
    stats = {}
    
    # 1. Overall win/loss ratio
    win_counts = combat_data['Winner'].value_counts()
    total_battles = len(combat_data)
    stats['overall'] = {winner: f"{count} ({count/total_battles*100:.1f}%)"
                       for winner, count in win_counts.items()}
    
    # 2. HP difference analysis
    combat_data['HP_Diff'] = combat_data['HP1'] - combat_data['HP2']
    
    # Divide HP difference into 4 categories using fixed ranges
    hp_diff_mean = combat_data['HP_Diff'].mean()
    hp_diff_std = combat_data['HP_Diff'].std()
    
    def get_hp_category(diff):
        if diff < hp_diff_mean - hp_diff_std:
            return 'Very Disadvantaged'
        elif diff < hp_diff_mean:
            return 'Disadvantaged'
        elif diff < hp_diff_mean + hp_diff_std:
            return 'Advantaged'
        else:
            return 'Very Advantaged'
    
    combat_data['HP_Category'] = combat_data['HP_Diff'].apply(get_hp_category)
    
    stats['hp_analysis'] = {}
    for category in ['Very Disadvantaged', 'Disadvantaged', 'Advantaged', 'Very Advantaged']:
        category_data = combat_data[combat_data['HP_Category'] == category]
        if len(category_data) > 0:
            win_ratio = category_data['Winner'].value_counts(normalize=True)
            stats['hp_analysis'][category] = {
                'count': len(category_data),
                'win_ratio': {winner: f"{ratio*100:.1f}%" for winner, ratio in win_ratio.items()}
            }
    
    # 3. Attack/Defense efficiency analysis
    combat_data['Attack_Efficiency1'] = combat_data['Attack1'] / combat_data['Defense2']
    combat_data['Attack_Efficiency2'] = combat_data['Attack2'] / combat_data['Defense1']
    stats['efficiency'] = {
        'PC1_wins': {
            'avg_attack_eff': combat_data[combat_data['Winner'] == 'PC1']['Attack_Efficiency1'].mean(),
            'avg_defense_eff': 1/combat_data[combat_data['Winner'] == 'PC1']['Attack_Efficiency2'].mean()
        },
        'PC2_wins': {
            'avg_attack_eff': combat_data[combat_data['Winner'] == 'PC2']['Attack_Efficiency2'].mean(),
            'avg_defense_eff': 1/combat_data[combat_data['Winner'] == 'PC2']['Attack_Efficiency1'].mean()
        }
    }
    
    # 4. Combat time analysis
    stats['time_analysis'] = {
        'avg_time': combat_data['Total_Time'].mean(),
        'by_winner': {winner: group['Total_Time'].mean() 
                     for winner, group in combat_data.groupby('Winner')}
    }
    
    return stats

class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Training Progress')
        self.setGeometry(100, 900, 1200, 400)  # Position below main window
        
        # Central widget setup
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Loss graph figure
        self.loss_figure, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.loss_canvas = FigureCanvas(self.loss_figure)
        layout.addWidget(self.loss_canvas)
    
    def update_plots(self, loss_data):
        """Update loss and accuracy plots during training"""
        try:
            epochs = range(1, loss_data['current_epoch'] + 1)
            
            # Combat Time Loss plot
            self.ax1.clear()
            self.ax1.plot(epochs, loss_data['time_loss'], 'b-', label='Training Loss')
            self.ax1.plot(epochs, loss_data['time_val_loss'], 'r-', label='Validation Loss')
            self.ax1.set_title('Combat Time Prediction Loss')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss (MSE)')
            self.ax1.legend()
            self.ax1.grid(True)
            
            # Combat Time MAE plot
            self.ax2.clear()
            self.ax2.plot(epochs, loss_data['time_mae'], 'b-', label='Training MAE')
            self.ax2.plot(epochs, loss_data['time_val_mae'], 'r-', label='Validation MAE')
            self.ax2.set_title('Combat Time Mean Absolute Error')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('MAE')
            self.ax2.legend()
            self.ax2.grid(True)
            
            # Winner Prediction Loss plot
            self.ax3.clear()
            self.ax3.plot(epochs, loss_data['winner_loss'], 'b-', label='Training Loss')
            self.ax3.plot(epochs, loss_data['winner_val_loss'], 'r-', label='Validation Loss')
            self.ax3.set_title('Winner Prediction Loss')
            self.ax3.set_xlabel('Epoch')
            self.ax3.set_ylabel('Loss (Cross-Entropy)')
            self.ax3.legend()
            self.ax3.grid(True)
            
            # Winner Prediction Accuracy plot
            self.ax4.clear()
            self.ax4.plot(epochs, loss_data['winner_accuracy'], 'b-', label='Training Accuracy')
            self.ax4.plot(epochs, loss_data['winner_val_accuracy'], 'r-', label='Validation Accuracy')
            self.ax4.set_title('Winner Prediction Accuracy')
            self.ax4.set_xlabel('Epoch')
            self.ax4.set_ylabel('Accuracy')
            self.ax4.legend()
            self.ax4.grid(True)
            
            self.loss_figure.tight_layout()
            self.loss_canvas.draw()
        except Exception as e:
            print(f"Error updating loss plots: {str(e)}")

class CombatBalanceAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Combat Balance Analyzer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create training progress window
        self.training_window = TrainingWindow()
        
        # Main widget and layout setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel (Controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Log display widget
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(150)
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # PC1 data group
        pc1_group = QGroupBox("PC1 Data")
        pc1_layout = QVBoxLayout()
        self.load_pc1_btn = QPushButton('Select PC1 CSV File')
        self.load_pc1_btn.clicked.connect(self.load_pc1_csv)
        self.pc1_stats = QTextEdit()
        self.pc1_stats.setReadOnly(True)
        self.pc1_stats.setMaximumHeight(100)
        pc1_layout.addWidget(self.load_pc1_btn)
        pc1_layout.addWidget(QLabel('PC1 Data Statistics:'))
        pc1_layout.addWidget(self.pc1_stats)
        pc1_group.setLayout(pc1_layout)
        left_layout.addWidget(pc1_group)
        
        # PC2 data group
        pc2_group = QGroupBox("PC2 Data")
        pc2_layout = QVBoxLayout()
        self.load_pc2_btn = QPushButton('Select PC2 CSV File')
        self.load_pc2_btn.clicked.connect(self.load_pc2_csv)
        self.pc2_stats = QTextEdit()
        self.pc2_stats.setReadOnly(True)
        self.pc2_stats.setMaximumHeight(100)
        pc2_layout.addWidget(self.load_pc2_btn)
        pc2_layout.addWidget(QLabel('PC2 Data Statistics:'))
        pc2_layout.addWidget(self.pc2_stats)
        pc2_group.setLayout(pc2_layout)
        left_layout.addWidget(pc2_group)
        
        # Training configuration group
        training_config_group = QGroupBox("Training Settings")
        training_config_layout = QVBoxLayout()
        
        # Epochs input
        epochs_layout = QHBoxLayout()
        self.epochs_input = QLineEdit()
        self.epochs_input.setPlaceholderText('100')
        epochs_layout.addWidget(QLabel('Number of Epochs:'))
        epochs_layout.addWidget(self.epochs_input)
        training_config_layout.addLayout(epochs_layout)
        
        # Optimization steps input
        steps_layout = QHBoxLayout()
        self.steps_input = QLineEdit()
        self.steps_input.setPlaceholderText('100')
        steps_layout.addWidget(QLabel('Optimization Steps:'))
        steps_layout.addWidget(self.steps_input)
        training_config_layout.addLayout(steps_layout)
        
        training_config_group.setLayout(training_config_layout)
        left_layout.addWidget(training_config_group)
        
        # Start training button
        self.train_btn = QPushButton('Start Model Training')
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        left_layout.addWidget(self.train_btn)
        
        # Target combat time input
        time_input_layout = QHBoxLayout()
        self.target_time_input = QLineEdit()
        self.target_time_input.setPlaceholderText('15.0')
        time_input_layout.addWidget(QLabel('Target Combat Time:'))
        time_input_layout.addWidget(self.target_time_input)
        left_layout.addLayout(time_input_layout)
        
        # Optimization button
        self.optimize_btn = QPushButton('Optimize Character Stats')
        self.optimize_btn.clicked.connect(self.optimize_character)
        self.optimize_btn.setEnabled(False)
        left_layout.addWidget(self.optimize_btn)
        
        # Optimization results display
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        left_layout.addWidget(QLabel('Optimization Results:'))
        left_layout.addWidget(self.result_display)
        
        # Right panel (Graphs)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Matplotlib canvas for analysis graphs
        self.figure, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        # Add panels
        layout.addWidget(left_panel, stretch=30)
        layout.addWidget(right_panel, stretch=70)
        
        # Initial state setup
        self.model = None
        self.history = None
        self.pc1_data = None
        self.pc2_data = None
        self.combat_data = None
        self.training_thread = None
        self.optimization_thread = None
        self.progress_dialog = None
        
        self.init_ui()
    
    def closeEvent(self, event):
        # Close training window
        self.training_window.close()
        
        # Clean up threads
        if hasattr(self, 'training_thread') and self.training_thread:
            if self.training_thread.isRunning():
                self.training_thread.terminate()
                self.training_thread.wait()
            self.training_thread.cleanup()
            
        if hasattr(self, 'optimization_thread') and self.optimization_thread:
            if self.optimization_thread.isRunning():
                self.optimization_thread.terminate()
                self.optimization_thread.wait()
        
        # Clean up tensorflow session
        tf.keras.backend.clear_session()
        
        event.accept()
    
    def init_ui(self):
        # 초기 상태 설정
        self.model = None
        self.history = None
        self.pc1_data = None
        self.pc2_data = None
        self.combat_data = None
        self.training_thread = None
        self.optimization_thread = None
        self.progress_dialog = None
    
    def show_progress_dialog(self, title, text):
        self.progress_dialog = QProgressDialog(text, None, 0, 100, self)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.show()
        
    def update_progress(self, value):
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setValue(value)
        
    def load_pc1_csv(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(self, 'PC1 Combat Data CSV File Selection', '', 'CSV files (*.csv)')
            if filename:
                # Use the first row as headers and load data
                data = pd.read_csv(filename, header=0)
                
                # Check for required columns
                required_columns = ["HP1", "Attack1", "Defense1"]
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    QMessageBox.critical(self, 'Error', 
                        f'Missing required columns in CSV file:\n{", ".join(missing_columns)}\n\n'
                        f'Required column list:\n{", ".join(required_columns)}')
                    return
                
                # Convert data types
                for col in required_columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except Exception as e:
                        QMessageBox.critical(self, 'Error', f'Data type conversion error ({col}): {str(e)}')
                        return
                
                # Remove rows with NaN values
                data = data.dropna(subset=required_columns)
                
                self.pc1_data = data
                self.pc1_stats.setText(str(self.pc1_data.describe()))
                self.check_data_ready()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error loading PC1 file: {str(e)}')
    
    def load_pc2_csv(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(self, 'PC2 Combat Data CSV File Selection', '', 'CSV files (*.csv)')
            if filename:
                # Use the first row as headers and load data
                data = pd.read_csv(filename, header=0)
                
                # Check for required columns
                required_columns = ["HP2", "Attack2", "Defense2"]
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    QMessageBox.critical(self, 'Error', 
                        f'Missing required columns in CSV file:\n{", ".join(missing_columns)}\n\n'
                        f'Required column list:\n{", ".join(required_columns)}')
                    return
                
                # Convert data types
                for col in required_columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except Exception as e:
                        QMessageBox.critical(self, 'Error', f'Data type conversion error ({col}): {str(e)}')
                        return
                
                # Remove rows with NaN values
                data = data.dropna(subset=required_columns)
                
                self.pc2_data = data
                self.pc2_stats.setText(str(self.pc2_data.describe()))
                self.check_data_ready()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error loading PC2 file: {str(e)}')
    
    def check_data_ready(self):
        if self.pc1_data is not None and self.pc2_data is not None:
            # Check if PC1 and PC2 data have the required columns
            pc1_required = ["HP1", "Attack1", "Defense1"]
            pc2_required = ["HP2", "Attack2", "Defense2"]
            
            pc1_missing = [col for col in pc1_required if col not in self.pc1_data.columns]
            pc2_missing = [col for col in pc2_required if col not in self.pc2_data.columns]
            
            if pc1_missing or pc2_missing:
                self.train_btn.setEnabled(False)
                return
            
            # If all conditions are met, enable training button
            self.train_btn.setEnabled(True)
    
    def start_training(self):
        try:
            if self.pc1_data is None or self.pc2_data is None:
                QMessageBox.critical(self, 'Error', 'Please load both PC1 and PC2 data.')
                return
            
            # Show training window
            self.training_window.show()
            
            # Check data size
            pc1_rows = len(self.pc1_data)
            pc2_rows = len(self.pc2_data)
            if pc1_rows != pc2_rows:
                QMessageBox.critical(self, 'Error', 
                    f'PC1 and PC2 data row count do not match.\n'
                    f'PC1: {pc1_rows} rows, PC2: {pc2_rows} rows')
                return
            
            # Merge data
            try:
                # Select only necessary columns and copy
                pc1_required = ["HP1", "Attack1", "Defense1"]
                pc2_required = ["HP2", "Attack2", "Defense2"]
                
                pc1_data = self.pc1_data[pc1_required].copy()
                pc2_data = self.pc2_data[pc2_required].copy()
                
                # Check data type conversion
                for col in pc1_required + pc2_required:
                    data = pc1_data if col in pc1_required else pc2_data
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Remove NaN values
                pc1_data = pc1_data.dropna()
                pc2_data = pc2_data.dropna()
                
                if len(pc1_data) != len(pc2_data):
                    QMessageBox.critical(self, 'Error', 
                        f'Valid data row count do not match.\n'
                        f'PC1: {len(pc1_data)} rows, PC2: {len(pc2_data)} rows')
                    return
                
                # Merge data
                self.combat_data = pd.concat([pc1_data, pc2_data], axis=1)
                
                # Final data check
                if len(self.combat_data) == 0:
                    QMessageBox.critical(self, 'Error', 'No valid data found.')
                    return
                
                self.log_message(f"Data processing completed: {len(self.combat_data)} rows")
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error merging data: {str(e)}')
                return
            
            # Get training epochs
            try:
                epochs = int(self.epochs_input.text() or 100)
                if epochs <= 0:
                    raise ValueError("Epochs must be a positive number.")
            except ValueError as e:
                QMessageBox.critical(self, 'Error', f'Invalid epochs: {str(e)}')
                return
            
            # Start training
            self.show_progress_dialog('Model Training in Progress', 'Training combat simulation and neural network model...')
            
            self.training_thread = TrainingThread(self.combat_data.copy(), epochs)
            self.training_thread.progress.connect(self.update_progress)
            self.training_thread.finished.connect(self.training_finished)
            self.training_thread.error.connect(self.handle_training_error)
            self.training_thread.loss_update.connect(self.update_loss_plot)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error occurred during model training: {str(e)}')
    
    def training_finished(self, result):
        if result:
            try:
                self.model = result['model']
                self.history = result['history']
                self.combat_data = result['combat_data']
                self.X_mean = result['X_mean']
                self.X_std = result['X_std']
                
                # 회귀 분석을 위한 예측값 생성
                X_normalized = (self.combat_data[['HP1', 'Attack1', 'Defense1', 'HP2', 'Attack2', 'Defense2']] - self.X_mean) / self.X_std
                predictions = self.model.predict(X_normalized)
                predicted_times = predictions[0].flatten()
                actual_times = self.combat_data['Total_Time']
                
                # R² 값 계산
                correlation_matrix = np.corrcoef(actual_times, predicted_times)
                r_squared = correlation_matrix[0, 1] ** 2
                
                # 예측값을 데이터프레임에 저장
                self.combat_data['Predicted_Time'] = predicted_times
                
                self.optimize_btn.setEnabled(True)
                
                # Perform balance analysis
                stats = analyze_balance_stats(self.combat_data)
                
                # Display statistics
                stats_text = "🎮 Combat Balance Analysis Results\n\n"
                
                # 1. Overall win/loss ratio
                stats_text += "1. Overall Win/Loss Ratio:\n"
                for winner, ratio in stats['overall'].items():
                    stats_text += f"   - {winner}: {ratio}\n"
                
                # 2. Win/Loss Analysis by HP Difference
                stats_text += "\n2. Win/Loss Analysis by HP Difference:\n"
                for condition, data in stats['hp_analysis'].items():
                    stats_text += f"   - {condition} (Total {data['count']} battles):\n"
                    for winner, ratio in data['win_ratio'].items():
                        stats_text += f"     * {winner}: {ratio}\n"
                
                # 3. Attack/Defense Efficiency Analysis
                stats_text += "\n3. Attack/Defense Efficiency Analysis:\n"
                for winner, effs in stats['efficiency'].items():
                    stats_text += f"   - When {winner} wins:\n"
                    stats_text += f"     * Avg Attack Efficiency: {effs['avg_attack_eff']:.2f}\n"
                    stats_text += f"     * Avg Defense Efficiency: {effs['avg_defense_eff']:.2f}\n"
                
                # 4. Combat Time Analysis
                stats_text += "\n4. Combat Time Analysis:\n"
                stats_text += f"   - Average Combat Time: {stats['time_analysis']['avg_time']:.1f} turns\n"
                for winner, avg_time in stats['time_analysis']['by_winner'].items():
                    stats_text += f"   - {winner} Victory Average: {avg_time:.1f} turns\n"
                
                # 5. Regression Analysis
                stats_text += f"\n5. Regression Analysis:\n"
                stats_text += f"   - R² Score: {r_squared:.4f}\n"
                stats_text += f"   - Mean Absolute Error: {np.mean(np.abs(actual_times - predicted_times)):.2f} turns\n"
                
                # Balance evaluation
                win_counts = self.combat_data['Winner'].value_counts()
                total_battles = len(self.combat_data)
                pc1_ratio = win_counts.get('PC1', 0) / total_battles
                
                if abs(pc1_ratio - 0.5) < 0.1:
                    balance_evaluation = "🟢 현재 밸런스가 매우 균형잡혀 있습니다."
                elif abs(pc1_ratio - 0.5) < 0.2:
                    balance_evaluation = "🟡 밸런스가 약간 한쪽으로 치우쳐 있으나, 수용 가능한 수준입니다."
                else:
                    balance_evaluation = "🔴 밸런스가 크게 한쪽으로 치우쳐 있어 조정이 필요합니다."
                
                stats_text += f"\n💫 밸런스 평가:\n{balance_evaluation}"
                
                # Display results
                self.result_display.setText(stats_text)
                
                # Update graphs
                self.update_plots()
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error processing results: {str(e)}')
        else:
            QMessageBox.critical(self, 'Error', 'Error occurred during model training.')
    
    def update_loss_plot(self, loss_data):
        """Update loss and accuracy plots during training"""
        self.training_window.update_plots(loss_data)
        if not self.training_window.isVisible():
            self.training_window.show()
    
    def update_plots(self):
        try:
            if self.combat_data is None or len(self.combat_data) == 0:
                return
            
            # Clear all plots
            for ax in self.axes.flat:
                ax.clear()
            
            try:
                # 1. Win/Loss ratio pie chart
                win_counts = self.combat_data['Winner'].value_counts()
                self.axes[0, 0].pie(win_counts.values, labels=win_counts.index, autopct='%1.1f%%')
                self.axes[0, 0].set_title('Win/Loss Ratio')
            except Exception as e:
                print(f"Error creating pie chart: {str(e)}")
            
            try:
                # 2. Regression plot (Actual vs Predicted Times)
                actual_times = self.combat_data['Total_Time']
                predicted_times = self.combat_data['Predicted_Time']
                
                # Calculate regression line
                z = np.polyfit(actual_times, predicted_times, 1)
                p = np.poly1d(z)
                
                # Calculate R-squared
                correlation_matrix = np.corrcoef(actual_times, predicted_times)
                r_squared = correlation_matrix[0, 1] ** 2
                
                # Create scatter plot
                self.axes[0, 1].scatter(actual_times, predicted_times, alpha=0.5)
                self.axes[0, 1].plot(actual_times, p(actual_times), "r--", alpha=0.8)
                
                # Add perfect prediction line (y=x)
                min_val = min(actual_times.min(), predicted_times.min())
                max_val = max(actual_times.max(), predicted_times.max())
                self.axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                self.axes[0, 1].set_title(f'Actual vs Predicted Combat Time\nR² = {r_squared:.4f}')
                self.axes[0, 1].set_xlabel('Actual Combat Time')
                self.axes[0, 1].set_ylabel('Predicted Combat Time')
                self.axes[0, 1].grid(True)
            except Exception as e:
                print(f"Error creating regression plot: {str(e)}")
            
            try:
                # 3. Win rate by HP difference
                hp_diff_mean = self.combat_data['HP1'].mean() - self.combat_data['HP2'].mean()
                hp_diff_std = (self.combat_data['HP1'] - self.combat_data['HP2']).std()
                
                categories = ['Very Disadvantaged', 'Disadvantaged', 'Advantaged', 'Very Advantaged']
                win_ratios = []
                counts = []
                
                for category in categories:
                    if category == 'Very Disadvantaged':
                        mask = (self.combat_data['HP1'] - self.combat_data['HP2']) < (hp_diff_mean - hp_diff_std)
                    elif category == 'Disadvantaged':
                        mask = ((self.combat_data['HP1'] - self.combat_data['HP2']) >= (hp_diff_mean - hp_diff_std)) & \
                              ((self.combat_data['HP1'] - self.combat_data['HP2']) < hp_diff_mean)
                    elif category == 'Advantaged':
                        mask = ((self.combat_data['HP1'] - self.combat_data['HP2']) >= hp_diff_mean) & \
                              ((self.combat_data['HP1'] - self.combat_data['HP2']) < (hp_diff_mean + hp_diff_std))
                    else:  # Very Advantaged
                        mask = (self.combat_data['HP1'] - self.combat_data['HP2']) >= (hp_diff_mean + hp_diff_std)
                    
                    category_data = self.combat_data[mask]
                    if len(category_data) > 0:
                        win_ratio = (category_data['Winner'] == 'PC1').mean() * 100
                        win_ratios.append(win_ratio)
                        counts.append(len(category_data))
                    else:
                        win_ratios.append(0)
                        counts.append(0)
                
                # Draw win rate graph
                bars = self.axes[1, 0].bar(categories, win_ratios)
                self.axes[1, 0].set_title('PC1 Win Rate by HP Difference')
                self.axes[1, 0].set_ylabel('Win Rate (%)')
                self.axes[1, 0].set_ylim(0, 100)
                
                # Show data count on each bar
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    self.axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                       f'n={count}',
                                       ha='center', va='bottom')
            
            except Exception as e:
                print(f"Error creating HP difference graph: {str(e)}")
            
            try:
                # 4. Prediction Error Distribution
                errors = self.combat_data['Predicted_Time'] - self.combat_data['Total_Time']
                self.axes[1, 1].hist(errors, bins=30, alpha=0.7)
                self.axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
                self.axes[1, 1].set_title('Prediction Error Distribution')
                self.axes[1, 1].set_xlabel('Prediction Error (Predicted - Actual)')
                self.axes[1, 1].set_ylabel('Frequency')
                self.axes[1, 1].grid(True)
                
                # Add mean and std annotations
                mean_error = errors.mean()
                std_error = errors.std()
                self.axes[1, 1].text(0.02, 0.95, 
                                   f'Mean Error: {mean_error:.2f}\nStd Dev: {std_error:.2f}',
                                   transform=self.axes[1, 1].transAxes,
                                   bbox=dict(facecolor='white', alpha=0.8))
            except Exception as e:
                print(f"Error creating error distribution plot: {str(e)}")
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
    
    def optimize_character(self):
        try:
            target_time = float(self.target_time_input.text() or 15.0)
            steps = int(self.steps_input.text() or 100)
            
            # Start optimization
            self.show_progress_dialog('Optimization in Progress', 'Optimizing character stats...')
            
            self.optimization_thread = OptimizationThread(self.model, target_time, steps)
            self.optimization_thread.progress.connect(self.update_progress)
            self.optimization_thread.finished.connect(self.optimization_finished)
            self.optimization_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error occurred during optimization: {str(e)}')
    
    def optimization_finished(self, result):
        if result:
            final_result = (
                f"✅ Optimization Completed!\n"
                f"Result for Target Combat Time ({float(self.target_time_input.text() or 15.0)} seconds):\n"
                f"Attack: {result['attack']:.2f}\n"
                f"Defense: {result['defense']:.2f}\n\n"
                f"Optimization Process:\n{''.join(result['results'])}"
            )
            self.result_display.setText(final_result)
        else:
            QMessageBox.critical(self, 'Error', 'Error occurred during optimization.')

    def handle_training_error(self, error_message):
        """Display error message in log and show error dialog"""
        self.log_message(f"Error: {error_message}")
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.close()
        QMessageBox.critical(self, 'Error', error_message)

    def log_message(self, message):
        """Add log message to log display widget"""
        self.log_display.append(message)
        # Scroll to bottom
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CombatBalanceAnalyzer()
    window.show()
    sys.exit(app.exec_()) 