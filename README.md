## Описание модели 
Модель содержит веса $W_1$, $b_1$, $W_2$, $b_2$ в файле ```model_weights.npz```.
- $W_1 \in\mathbb{R}^{8 \times 2}$
- $b_1 \in\mathbb{R}^{8 \times 1}$
- $W_2 \in\mathbb{R}^{1 \times 8}$
- $b_2 \in\mathbb{R}$

На вход подается $X = (x_1, x_2)^T$
Состоит из слоёв:
1) Первый слой:  
   $z_1 = W_1 X + b_1$  
   $A_1 = ReLU(Z_1)$
2) Второй слой:  
   $z_2 = W_2 A_1 + b_2$  
   $A_2=\sigma(z_2)$  
   $\hat{y} = A_2$ — выход. 

Модель:

$$\hat{y}(x) = \sigma(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2)$$

состоит из композиции линейных и кусочно-гладких функций, представляет собой кусочно-гладкую функцию.  

В программе на вход подается массив из $X$ длиной $500$ из файла тестовой выборки ```data.csv```. Также в этом файле содержатся верные значения для предсказаний: $y \in\mathbb{R}^{500}$


 ## Функция потерь
 Для бинарной классификации подойдет loss-функция Binary Cross Entropy:
 
 $$L=-(ylog(\hat{y}) + (1-y)log(1-\hat{y}))$$
 
 Для массива из $n = 500$ объектов она примет вид:
 
 $$L = -\frac{1}{n}\sum_{i=1}^{n} (y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i))$$
 
 ## Аналитический метод поиска градиента
 Производная по $\hat{y}$ (скаляр):
 
 $$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}}+\frac{1-y}{1-\hat{y}}$$
 
 Производная сигмоиды:
 
 $$\frac{\partial \hat{y}}{\partial z_2}=\hat{y}(1-\hat{y})$$
 
 $$\frac{\partial L}{\partial z_2} =  \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} = \hat{y}-y$$
 
 Производная $z_2$ по $W_2:$  
 
 $z_2 = W_2 A_1 + b_2$  
 
 $W_2 \in\mathbb{R}^{1 \times 8}, A_1 \in\mathbb{R}^{8 \times 1}$, $b_2 \in\mathbb{R}$  
 
 $z_2 = W_2 A_1+b_2=\sum_{i=1}^{8}w_{2i}a_{i}+b_2$  
 
 $$\frac{\partial z_2}{\partial (W_2)_{i}} = a_{i}$$  
 
 $$\frac{\partial z_2}{\partial W_2} = A_1^T$$
 
 $z_2-$  скалярная функция:
 
 $$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_2} = (\hat{y}-y) \cdot A_1^T$$
 
 Производная по $b_2:$  
 
 $b_2 -$ скаляр, поэтому:
 
  $$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial b_2} = \frac{\partial L}{\partial z_2} \cdot 1 = (\hat{y}-y) $$
  
 далее
 
$$\frac{\partial L}{\partial A_1} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial A_1}$$

$$\frac{\partial z_2}{\partial (A_1)_{i}} = w_{2i}$$
 
 $$\frac{\partial z_2}{\partial A_1}=W_2^T$$
 
 
 $$\frac{\partial L}{\partial A_1} =(\hat{y}-y) \cdot W_2^T$$
 
 Производные по $W_1$ и $b_1:$  
 
 $a_i = ReLU((z_1)_i) = max(0, (z_1)_i)$
 
  $$\frac{\partial a_i}{\partial (z_1)_i} = \begin{cases} 0, & (z_1)_i \le 0 \\ 1, & (z_1)_i > 0 \end {cases}$$
  
  
  $$\frac{\partial L}{\partial (b_1)_i} = \frac{\partial L}{\partial (z_1)_i} \cdot \frac{\partial (z_1)_i}{\partial (b_1)_i}= \frac{\partial L}{\partial (z_1)_i} \cdot 1=\frac{\partial L}{\partial (A_1)_i} \cdot \frac{\partial a_i}{\partial (z_1)_i} = w_{2i} \cdot \frac{\partial a_i}{\partial (z_1)_i}$$
  $z_1 = W_1 X + b_1$
  $(z_1)_i=\sum_{j=1}^{2} (W_1)_{ij} \cdot x_j + (b_1)_i$
  $$\frac{\partial L}{\partial (W_1)_{ij}} = \sum_{k=1}^{8} \frac{\partial L}{\partial (z_1)_k} \cdot \frac{\partial (z_1)_k}{\partial (W_1)_{ij}}$$
  $$\frac{\partial (z_1)_i}{\partial (W_1)_{ij}} = x_j, $$ если $k = i$, иначе $0$
   $$\frac{\partial L}{\partial (W_1)_{ij}} = \frac{\partial L}{\partial (z_1)_i} \cdot x_j = w_{2i} \cdot \frac{\partial a_i}{\partial (z_1)_i} \cdot x_j$$

#### Для батча из $n = 500$ элементов 

$$\begin{aligned}
\frac {\partial L}{\partial z_2} &= \hat{y} - y \in\mathbb{R}^{1 \times n},  
\frac{\partial L}{\partial W_2} &= \frac{1}{n} \cdot \frac {\partial L}{\partial z_2} \cdot A_1^T \in\mathbb{R}^{1 \times 8},  
\frac{\partial L}{\partial b_2} &= \frac{1}{n} \cdot \frac {\partial L}{\partial z_2} \cdot \mathbf{1}_{n \times 1} \in\mathbb{R}^{1 \times 1},
\frac{\partial L}{\partial A_1} &= W_2^T \cdot \frac {\partial L}{\partial z_2}\in\mathbb{R}^{8 \times n},
\frac{\partial L}{\partial z_1} &= \frac{\partial L}{\partial A_1} \odot \mathbb{1}_{(z_1 > 0)} \in\mathbb{R}^{8 \times n},
\frac{\partial L}{\partial W_1} &= \frac{1}{n} \cdot \frac{\partial L}{\partial z_1} \cdot X \in\mathbb{R}^{8 \times 2},
\frac{\partial L}{\partial b_1} &= \frac{1}{n} \cdot \frac{\partial L}{\partial z_1} \cdot \mathbf{1}_{n \times 1} \in\mathbb{R}^{8 \times 1}
\end{aligned}$$

## Численный метод (Метод конечных разностей)

$$\frac{\partial L}{\partial c_i} \approx \frac{L(c_i + \varepsilon) - L(c_i - \varepsilon)}{2\varepsilon}, $$ шаг $\quad \varepsilon = 10^{-5}$  
$c = W_1 |b_1|W_2 |b_2$  
Для проверки будем использовать относительную разность, так как градиенты могут иметь разный масштаб, а она их нормирует:
$$\text{reldiff} = \max_i \frac{|grad_{\text{num}}^{(i)} - grad_{\text{an}}^{(i)}|}{|grad_{\text{num}}^{(i)}| + |grad_{\text{an}}^{(i)}| }$$

### Результаты градиентной проверки
На случайных 50 входных данных погрешность не больше $10 ^{-4}$

| Параметр | Относительная ошибка | Статус |
|----------|---------------------|--------|
| $W_1$ | $3.09 \times 10^{-9}$ | PASS |
| $b_1$ | $1.87 \times 10^{-10}$ | PASS |
| $W_2$ | $4.45 \times 10^{-10}$ | PASS |
| $b_2$ | $3.43 \times 10^{-12}$ | PASS |

## Устойчивость модели

Для каждой правильно классифицированной точки $X$ с истинной меткой $y$ необходимо найти минимальное возмущение $\delta$, такое что: 

$$\hat{y}(X+\delta) \neq y$$

Для точки $X$ с истинной меткой $y$ определим функцию потерь

 $$L(X,y)=-(ylog(\hat{y}(X)) + (1-y)log(1-\hat{y}(X)))$$
 
 Нужно максимизировать $L(x, y)$ чтобы модель ошиблась. Для этого можно применить итеративный градиентный метод PGD:
 
 $$\delta_{t+1}=\delta_t+\alpha \cdot \nabla_X L(X+\delta_t,y)$$
 
 где: $\alpha-$шаг градиентного подъема, $t-$ номер итерации.  
 Идем вверх по градиенту функции потерь, поэтому для максимальной ошибки $\alpha > 0$.  
 Процесс останавливается при выполнении условия:
 
 $$\hat{y}(X+\delta_t) \neq y$$
 
Полученное $\delta = \delta_t$ является искомым возмущением.  
Градиент по X:

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial X}$$

$\frac{\partial z_1}{\partial X} =W_1^T$

$$\nabla_X L(X, y) = W_1^T \left( \left( W_2^T (\hat{y} - y) \right) \odot \mathbb{1}_{(z_1 > 0)} \right)$$

Для бинарной классификации с сигмоидой решение модели определяется знаком $z_2(X):$ 

$$\hat{y}(X) = \sigma (z_2(X)) > 0.5\Leftrightarrow z_2>0$$

Поэтому можно искать градиент $z_2$ по $X:$

$$\nabla_X L = \frac{\partial L}{\partial z_2} \cdot \nabla_X z_2 = (\hat{y} - y) \cdot \nabla_X z_2$$

$$\frac{\partial z_2}{\partial X} = \frac{\partial z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial X}$$

$$\frac{\partial z_2}{\partial X} = \left( W_2^T \odot \mathbb{1}_{(z_1 > 0)} \right)^T \cdot W_1^T$$

В реализации попроще: если истинная метка равна 1, то двигаемся против градиента, иначе по нему.
Для кусочно-линейной функции и старта из 0 — движение по градиенту ведет к ближайшей границе, поэтому полученное возмущение будет близко к минимальному по норме 2.
### Недостаток алгоритма 
Алгоритм PGD находит только локальные экстремумы. Поэтому можно его улучшить задавая случайные значения $\delta_0$, а не начинать с $\delta_0 = 0$. Так можно исследовать несколько областей, а потом сравнить значения и выбрать подходящее, это поможет алгоритму не застрять в одном месте.
Но так как требуется найти минимальное отклонение, то можно не усложнять.

## Анализ графика 
Был построен график ```adversarial_plot.png```, разделяющий точки двух классов. Нетрудно заметить, что наиболее уязвимые точки концентрируются около границы. Области с высокой плотностью точек разных классов также демонстрируют
повышенную уязвимость.
## Ответ на вопрос
Модель в этой точке может быть уверена в своем предсказании, но сама точка может находится близко к границе, из-за этого она неустойчива к возмущениям. Модель обучается минимизировать ошибку на обучающей выборке, а не максимизировать расстояние до границы.
Из-за ReLU граница становится кусочно-гладкой (как видно на графике - кусочно-линейной). В местах изломов точка может оказаться окружена границей с нескольких сторон, из-за чего ее можно "выбить" с разных направлений.

### Комментарий
При выполнении прямого прохода обученной модели возникали
предупреждения RuntimeWarning. Проверка
показала, что веса модели находятся в диапазоне $[-2.24, 2.70]$,
входные данные не содержат значений
$inf$ или $nan$. Предупреждения обусловлены особенностями
арифметики с плавающей точкой при промежуточных вычислениях
и не влияют на конечный результат.
