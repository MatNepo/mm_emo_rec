import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# === Настройки пользователя для наглядности ===
wav_path     = '../data/aud_from_vid/concert.wav'
start_sec    = 0.0      # начало фрагмента (с)
duration_sec = 0.3      # длина фрагмента (с)
kernel_size  = 11       # размер первого Conv1D ядра
stride1      = 5        # шаг первого Conv1D
block_k      = 3        # размер ядра в блоках
block_stride = 1        # шаг в блоках
pool_s       = 4        # размер окна пулинга
n_filters    = 16       # число фильтров (квадрат для наглядности)

# --- Загрузка и подготовка ---
sr, data = wavfile.read(wav_path)
if data.ndim > 1: data = data.mean(axis=1)
seg = data[int(start_sec*sr):int((start_sec+duration_sec)*sr)]

# --- Генерируем демонстрационные фильтры ---
np.random.seed(42)
f1 = np.random.randn(n_filters, kernel_size)
fb = [np.random.randn(n_filters, block_k) for _ in range(3)]

# --- Операции свёртки и пулинга ---
def conv1d(x, filt, stride):
    n_f, k = filt.shape
    L = (len(x) - k)//stride + 1
    out = np.zeros((n_f, L))
    for i in range(L):
        out[:,i] = filt.dot(x[i*stride:i*stride+k])
    return out

def pool1d(x, s):
    n_f, L = x.shape
    Lp = L//s
    x = x[:,:Lp*s]
    return x.reshape(n_f, Lp, s).max(axis=2)

# --- Пайплайн Wavegram ветки ---
out1 = conv1d(seg, f1, stride1)
p1   = pool1d(out1, pool_s)
out2 = conv1d(p1.flatten(), fb[0], block_stride)
p2   = pool1d(out2, pool_s)
out3 = conv1d(p2.flatten(), fb[1], block_stride)
p3   = pool1d(out3, pool_s)
out4 = conv1d(p3.flatten(), fb[2], block_stride)

# === НАГЛЯДНАЯ ВИЗУАЛИЗАЦИЯ ЭТАПОВ СВЁРТОК ===

# 1. Сырая волна и первый фильтр
fig, axs_grid = plt.subplots(2, 2, figsize=(14, 7))

axs_grid[0][0].plot(np.arange(len(seg))/sr, seg, color='gray')
axs_grid[0][0].set_title('Сырой сигнал (фрагмент)')
axs_grid[0][0].set_xlabel('Время, с')
axs_grid[0][0].set_ylabel('Амплитуда')
axs_grid[0][0].axvspan(0, 11/sr, color='orange', alpha=0.2, label='Первые 11 сэмплов')
axs_grid[0][0].legend()

axs_grid[0][1].plot(np.arange(11), seg[:11], marker='o', color='orange')
axs_grid[0][1].set_title('Первые 11 сэмплов')
axs_grid[0][1].set_xlabel('Сэмпл')
axs_grid[0][1].set_ylabel('Амплитуда')

axs_grid[1][0].plot(f1[0], marker='o', color='purple')
axs_grid[1][0].set_title('Первый фильтр (ядро)')
axs_grid[1][0].set_xlabel('Сэмпл ядра')
axs_grid[1][0].set_ylabel('Вес')

axs_grid[1][1].plot(seg[:30], color='gray', label='Сигнал')
for i in range(0, 30-kernel_size+1, stride1):
    axs_grid[1][1].axvspan(i, i+kernel_size-1, color='orange', alpha=0.2)
axs_grid[1][1].set_title('Фильтр скользит по сигналу (окна)')
axs_grid[1][1].set_xlabel('Сэмпл')
axs_grid[1][1].set_ylabel('Амплитуда')
plt.tight_layout()
plt.show()

# 2. Первая свёртка и пулинг
resp1 = conv1d(seg, f1, stride1)
fig, ax_pool = plt.subplots(figsize=(12, 4))
ax = ax_pool if not isinstance(ax_pool, np.ndarray) else ax_pool[0]
ax.plot(resp1[0], label='Отклик первого фильтра', color='blue')
window = pool_s
max_points_x = []
max_points_y = []
for i in range(0, len(resp1[0])//window):
    start = i*window
    end = start+window
    rect = plt.Rectangle((start, min(resp1[0])), window, max(resp1[0])-min(resp1[0]), color='orange', alpha=0.1)
    ax.add_patch(rect)
    window_vals = resp1[0][start:end]
    max_idx = np.argmax(window_vals) + start
    max_val = resp1[0][max_idx]
    ax.plot(max_idx, max_val, 'ro')
    max_points_x.append(max_idx)
    max_points_y.append(max_val)
ax.plot(max_points_x, max_points_y, 'r-', linewidth=2, label='Максимумы (пулинг)')
ax.set_title('Пулинг: выделение максимумов в каждом окне')
ax.set_xlabel('Позиция окна')
ax.set_ylabel('Отклик')
ax.legend()
plt.tight_layout()
plt.show()

# 3. Глубокие блоки: повторные свёртки и пулинг (для одного канала)
p1 = pool1d(resp1, pool_s)
out2 = conv1d(p1.flatten(), fb[0], block_stride)
p2 = pool1d(out2, pool_s)
out3 = conv1d(p2.flatten(), fb[1], block_stride)
p3 = pool1d(out3, pool_s)
out4 = conv1d(p3.flatten(), fb[2], block_stride)

fig, axs_grid2 = plt.subplots(4, 2, figsize=(14, 10))
ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31 = axs_grid2.flat
ax00.plot(resp1[0], color='blue'); ax00.set_title('Отклик 1-й свёртки')
ax01.plot(p1[0], color='red'); ax01.set_title('Пулинг 1')
ax10.plot(out2[0], color='blue'); ax10.set_title('Отклик 2-й свёртки')
ax11.plot(p2[0], color='red'); ax11.set_title('Пулинг 2')
ax20.plot(out3[0], color='blue'); ax20.set_title('Отклик 3-й свёртки')
ax21.plot(p3[0], color='red'); ax21.set_title('Пулинг 3')
ax30.plot(out4[0], color='blue'); ax30.set_title('Отклик 4-й свёртки (финал)')
ax31.axis('off')
for ax in axs_grid2.flat:
    if hasattr(ax, 'set_xlabel'):
        ax.set_xlabel('Позиция')
        ax.set_ylabel('Значение')
plt.tight_layout()
plt.show()

# 4. Вектор признаков и карта признаков (wavegram)
F = int(np.sqrt(n_filters))
C = n_filters // F
Tprime = out4.shape[1]
wavegram = out4.reshape(C, F, Tprime).transpose(2,0,1)

# 4.1 Вектор признаков для одного временного среза
feature_vec = out4[:, 0]  # первый временной срез
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.bar(np.arange(len(feature_vec)), feature_vec, color='purple')
ax1.set_title('Вектор признаков (1 временной срез)')
ax1.set_xlabel('Индекс признака')
ax1.set_ylabel('Значение')

# 4.2 Карта признаков (wavegram) для всего сигнала
wavegram_img = wavegram.reshape(wavegram.shape[0], -1).T  # (C*F, T')
im = ax2.imshow(wavegram_img, origin='lower', aspect='auto', cmap='seismic',
               vmin=-np.max(np.abs(wavegram)), vmax=np.max(np.abs(wavegram)),
               extent=[0, Tprime, 0, wavegram_img.shape[0]])
ax2.set_title('Карта признаков (wavegram)')
ax2.set_xlabel('Время (шаги после свёрток и пулинга)')
ax2.set_ylabel('Частотные бины и каналы')
fig.colorbar(im, ax=ax2, orientation='vertical', label='Активация')
plt.tight_layout()
plt.show()

# 5. Читаемая спектрограмма
f, t, Sxx = spectrogram(seg, sr, nperseg=256, noverlap=128)
Sxx_db = 10 * np.log10(Sxx + 1e-10)
plt.figure(figsize=(10, 4))
im = plt.imshow(Sxx_db, aspect='auto', origin='lower', 
                extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')
plt.title('Спектрограмма исходного сигнала', fontsize=16)
plt.xlabel('Время (с)', fontsize=12)
plt.ylabel('Частота (Гц)', fontsize=12)
cbar = plt.colorbar(im, label='Мощность (дБ)')
plt.tight_layout()
plt.show()
