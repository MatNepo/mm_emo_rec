import numpy as np

def format_predictions(scores, labels):
    # Если scores это словарь, преобразуем его в список
    if isinstance(scores, dict):
        scores_list = []
        labels_list = []
        for label in labels:
            if label in scores:
                scores_list.append(scores[label])
                labels_list.append(label)
        scores = np.array(scores_list)
        labels = labels_list
    
    # Сортируем предсказания по убыванию значений
    sorted_indexes = np.argsort(scores)[::-1]
    
    # Форматируем вывод построчно для топ-10 предсказаний
    result = []
    for k in range(min(10, len(scores))):
        label = labels[sorted_indexes[k]]
        score = scores[sorted_indexes[k]]
        result.append(f"{label}: {score:.3f}")
    
    return "\n".join(result) 