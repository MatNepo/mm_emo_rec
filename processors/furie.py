import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# === Настройки пользователя ===
wav_path    = '../data/aud_from_vid/hp_test.wav'  # путь к WAV-файлу
start_sec   = 3.0    # время начала фрагмента (в секундах)
duration_sec= 0.3    # длина фрагмента (в секундах)
channel_mode= 'both' # 'left', 'right', 'mix', 'both' — какой канал анализировать

max_duration_sec = 0.2  # максимальная длина анализируемого фрагмента (в секундах), None — без ограничения

window_ms = 25  # длина окна в миллисекундах
hop_ms    = 15  # шаг окна в миллисекундах

n_windows = 2  # количество отображаемых окон

# === Чтение и подготовка ===
sr, data = wavfile.read(wav_path)

# Обрезка по времени
start_idx = int(start_sec * sr)
if max_duration_sec is not None:
    end_idx = start_idx + int(max_duration_sec * sr)
else:
    end_idx = start_idx + int(duration_sec * sr)
data = data[start_idx:end_idx]

# Для корректного отображения времени на графиках
# duration_sec теперь равен длине реально анализируемого фрагмента
if max_duration_sec is not None:
    duration_sec = min(max_duration_sec, duration_sec)
else:
    duration_sec = len(data) / sr

# Обработка каналов
if data.ndim == 1:
    # моно
    mono = data
    channels = {'Mono': mono}
else:
    # стерео
    left  = data[:, 0]
    right = data[:, 1]
    mix   = ((left.astype(float) + right.astype(float)) / 2).astype(data.dtype)
    channels = {'Left': left, 'Right': right, 'Mix': mix}

# Выбираем данные для анализа FFT и спектрограммы
if channel_mode == 'both':
    if data.ndim > 1:
        plot_channels = ['Left', 'Right']
        fft_data = channels['Mix']
    else:
        plot_channels = ['Mono']
        fft_data = channels['Mono']
elif channel_mode == 'mix':
    if data.ndim > 1:
        plot_channels = ['Mix']
        fft_data = channels['Mix']
    else:
        plot_channels = ['Mono']
        fft_data = channels['Mono']
else:
    plot_channels = [channel_mode.capitalize()]
    fft_data = channels[plot_channels[0]]

time_axis = np.arange(len(data)) / sr

# Параметры окон
win_len = int(sr * window_ms / 1000)
hop_len = int(sr * hop_ms / 1000)
t0 = 0.0
t1 = hop_ms / 1000.0
t2 = 2 * hop_ms / 1000.0

# === 1.1 Разбиение на окна ===
plt.figure(figsize=(5, 2))
for name in plot_channels:
    if name == 'Mono':
        plt.plot(time_axis, channels[name])
    else:
        plt.plot(time_axis, channels[name], label=name)
plt.xlim(0, duration_sec)
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.title('1.1 Перекрывающиеся окна')
ax = plt.gca()
colors = ['C0', 'C1', 'C2']
for i in range(n_windows):
    start = i * hop_ms / 1000.0
    ax.axvspan(start, start + window_ms/1000, alpha=0.3, color=colors[i % len(colors)],
               label=f'Окно {i+1}: {int(start*1000)}–{int((start+window_ms/1000)*1000)}\u00a0ms')
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()

# === 1.2 FFT одного окна ===
frame = fft_data[:win_len]
freqs = np.fft.rfftfreq(win_len, d=1/sr)
spectrum = np.abs(np.fft.rfft(frame))

plt.figure(figsize=(5, 2))
plt.plot(freqs, spectrum, linewidth=1)
plt.xlim(0, sr/2)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title(f'1.2 БПФ окна {window_ms} мс')
plt.tight_layout()

# === 1.3 Спектрограмма ===
f, t, Sxx = spectrogram(
    fft_data,
    fs=sr,
    window='hann',
    nperseg=win_len,
    noverlap=win_len - hop_len,
    scaling='spectrum'
)

plt.figure(figsize=(5, 2))
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
plt.ylabel('Частота (Гц)')
plt.xlabel('Время (с)')
plt.title('1.3 Спектрограмма')
plt.colorbar(label='Мощность (дБ)')
plt.ylim(0, sr/2)
plt.tight_layout()

plt.show()
