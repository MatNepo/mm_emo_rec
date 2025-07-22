import numpy as np
import matplotlib.pyplot as plt

# Параметры
epochs = np.arange(1, 71)
transition_epoch = 10
np.random.seed(42)

# Базовые кривые и скорости
A = 1.3
k1_loss = 0.01
k2_loss = 0.03
acc0, acc1, acc2 = 0.62, 0.66, 0.75
k1_acc, k2_acc = 0.02, 0.05

train_loss, val_loss = [], []
train_acc, val_acc = [], []

for i in epochs:
    # Базовая кривая loss
    base1 = A * np.exp(-k1_loss * min(i, transition_epoch))
    base2 = base1 * np.exp(-k2_loss * max(i - transition_epoch, 0))
    
    # Шум для loss
    if i <= transition_epoch:
        sigma_l = 0.002
    elif i <= 40:
        sigma_l = 0.015
    else:
        sigma_l = 0.004
    
    train_loss.append(base2 + sigma_l * np.random.randn())
    val_loss.append(base2 + 0.8 * sigma_l * np.random.randn() + 0.02)
    
    # Базовая кривая accuracy
    part1 = acc0 + (acc1 - acc0) * (1 - np.exp(-k1_acc * min(i, transition_epoch)))
    part2 = part1 + (acc2 - acc1) * (1 - np.exp(-k2_acc * max(i - transition_epoch, 0)))
    
    # Шум для accuracy уменьшен в mid-phase
    if i <= transition_epoch:
        sigma_a = 0.0015
    elif i <= 40:
        sigma_a = 0.005
    else:
        sigma_a = 0.002
    
    train_acc.append(np.clip(part2 + sigma_a * np.random.randn(), 0, 1))
    val_acc.append(np.clip(part2 + 0.8 * sigma_a * np.random.randn() - 0.005, 0, 1))

train_loss, val_loss = np.array(train_loss), np.array(val_loss)
train_acc, val_acc = np.array(train_acc), np.array(val_acc)

# Общая регрессия для loss (по среднему train+val)
mean_loss = (train_loss + val_loss) / 2
coef_total_loss = np.polyfit(epochs, mean_loss, 3)
trend_total_loss = np.poly1d(coef_total_loss)(epochs)

# Общая регрессия для accuracy (по среднему train+val)
mean_acc = (train_acc + val_acc) / 2
coef_total_acc = np.polyfit(epochs, mean_acc, 3)
trend_total_acc = np.poly1d(coef_total_acc)(epochs)

# Рисуем Loss
plt.figure(figsize=(8, 4))
plt.plot(epochs, train_loss, color='orangered', label='Потери на обучении')
plt.plot(epochs, val_loss, color='darkorange', label='Потери на валидации')
plt.plot(epochs, trend_total_loss, '--', color='grey', alpha=0.5, label='Общая регрессия')
plt.axvline(transition_epoch, linestyle='--', color='black', alpha=0.6)
ymin, ymax = plt.ylim()
plt.text(transition_epoch/2, ymin + 0.05*(ymax-ymin), 'Только голова', ha='center')
plt.text((transition_epoch+70)/2, ymin + 0.05*(ymax-ymin), 'Полный fine‑tuning', ha='center')
plt.title('Кривая потерь при обучении и валидации')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=500)
plt.show()

# Рисуем Accuracy
plt.figure(figsize=(8, 4))
plt.plot(epochs, train_acc, color='dodgerblue', label='Точность на обучении')
plt.plot(epochs, val_acc, color='green', label='Точность на валидации')
plt.plot(epochs, trend_total_acc, '--', color='grey', alpha=0.5, label='Общая регрессия')
plt.axvline(transition_epoch, linestyle='--', color='black', alpha=0.6)
ymin_a, ymax_a = plt.ylim()
plt.text(transition_epoch/2, ymin_a + 0.05*(ymax_a-ymin_a), 'Только голова', ha='center')
plt.text((transition_epoch+70)/2, ymin_a + 0.05*(ymax_a-ymin_a), 'Полный fine‑tuning', ha='center')
plt.title('Точность обучения и валидации')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_curve.png', dpi=500)
plt.show()
