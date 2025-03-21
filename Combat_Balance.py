import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ 전투 로그 데이터 생성 (랜덤 데이터)
np.random.seed(42)  # 재현성을 위한 시드 설정

# 더 많은 데이터 포인트 생성 (50개)
n_samples = 50


combat_logs = pd.DataFrame({
    "HP1": np.random.randint(800, 1500, n_samples),  # HP 범위 확대
    "Attack1": np.random.randint(40, 80, n_samples),  # 공격력 범위 확대
    "Defense1": np.random.randint(15, 35, n_samples),  # 방어력 범위 확대
    "HP2": np.random.randint(800, 1500, n_samples),
    "Attack2": np.random.randint(40, 80, n_samples),
    "Defense2": np.random.randint(15, 35, n_samples),
    # 전투 시간은 스탯에 따라 더 다양한 범위로 설정
    "Total_Time": np.random.uniform(10.0, 20.0, n_samples)
})

# 데이터 확인을 위한 기본 통계 출력
print("📊 생성된 전투 데이터 통계:")
print(combat_logs.describe())
print("\n🎲 데이터 샘플 (처음 5개):")
print(combat_logs.head())

# 2️⃣ 독립 변수 (X)와 종속 변수 (Y) 분리
X = combat_logs.drop(columns=["Total_Time"])
y = combat_logs["Total_Time"]

# 3️⃣ 훈련 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ 신경망 모델 생성
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),  # 입력층
    layers.Dense(32, activation="relu"),  # 은닉층
    layers.Dense(1)  # 출력층 (전투 시간 예측)
])

# 5️⃣ 모델 컴파일
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 6️⃣ 모델 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))

# 학습 과정 시각화
plt.figure(figsize=(12, 4))

# 1. 학습 곡선
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 2. 실제값 vs 예측값 산점도
plt.subplot(1, 2, 2)
y_pred = model.predict(X)
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Combat Time')
plt.xlabel('Actual Time')
plt.ylabel('Predicted Time')

plt.tight_layout()
plt.show()

# 7️⃣ 목표 전투 시간을 위한 최적화 함수
def optimize_character(target_time=15.0):
    attack = tf.Variable(50.0, dtype=tf.float32)  # 초기 공격력
    defense = tf.Variable(20.0, dtype=tf.float32)  # 초기 방어력

    optimizer = tf.optimizers.Adam(learning_rate=0.1)

    for i in range(100):
        with tf.GradientTape() as tape:
            # 입력 데이터를 동적으로 구성
            input_data = tf.stack([
                tf.constant(1000.0),
                attack,
                defense,
                tf.constant(900.0),
                tf.constant(55.0),
                tf.constant(18.0)
            ])
            input_data = tf.reshape(input_data, [1, 6])  # 배치 차원 추가
            
            # 모델 예측
            pred_time = model(input_data, training=False)[0][0]
            
            # 손실 계산
            loss = tf.abs(pred_time - target_time)

        # 그래디언트 계산
        gradients = tape.gradient(loss, [attack, defense])
        
        # 그래디언트가 None이 아닌 경우에만 업데이트
        if gradients[0] is not None and gradients[1] is not None:
            optimizer.apply_gradients(zip(gradients, [attack, defense]))

        if i % 10 == 0:
            print(f"Step {i}: Attack = {attack.numpy():.2f}, Defense = {defense.numpy():.2f}, Predicted Time = {pred_time.numpy():.2f}")

    return attack.numpy(), defense.numpy()

# 8️⃣ 최적화 실행
optimized_attack, optimized_defense = optimize_character(target_time=15.0)
print(f"✅ 최적화 완료! 목표 전투 시간(15초)에 맞춘 공격력: {optimized_attack:.2f}, 방어력: {optimized_defense:.2f}")