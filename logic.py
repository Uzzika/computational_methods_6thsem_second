import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSpinBox,
                             QLabel, QMessageBox, QComboBox, QGroupBox, QFrame)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPalette

class FirepowerOptimizer:
    def __init__(self, C, k, m=1, r=1):
        self.C = np.array(C, dtype=int)  # Целочисленная матрица
        self.k = k
        self.m = m
        self.r = r
        self.n = len(C)

    def optimize(self):
        if self.m == 1 and self.r == 1:
            return self._solve_simple()
        elif self.m == 2 and self.r == 2:
            return self._solve_advanced()
        else:
            raise ValueError("Unsupported parameters")

    def _solve_simple(self):
        row_ind, col_ind = linear_sum_assignment(-self.C)
        sigma = col_ind.tolist()
        total_power = self.C.sum() - (self.k-1)/self.k * sum(self.C[sigma[j], j] for j in range(self.n))
        return [sigma], total_power

    def _solve_advanced(self):
        best_schedule = None
        min_power = np.inf
        
        # Эвристика: выбираем топ-m подразделений в каждом периоде
        attacked = np.zeros(self.n, dtype=int)
        schedule = []
        
        for j in range(self.n):
            available = [i for i in range(self.n) if attacked[i] < self.r]
            if len(available) < self.m:
                break
                
            targets = sorted(available, key=lambda x: -self.C[x, j])[:self.m]
            schedule.append(targets)
            for t in targets:
                attacked[t] += 1
        
        if schedule:
            power = self._calculate_power(schedule)
            return schedule, power
        return None, np.inf

    def _calculate_power(self, schedule):
        total = 0
        for j in range(len(schedule)):
            period_power = self.C[:, j].sum()
            attacked_power = sum(self.C[t, j] for t in schedule[j])
            total += period_power - (self.k-1)/self.k * attacked_power
        return total
    

class StyledButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setMinimumHeight(40)
        self.setFont(QFont("Arial", 10))
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)


class MatrixEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Оптимизатор огневой мощи")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                border-radius: 12px;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }
            QLabel {
                color: #333;
            }
            QTableWidget {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            QSpinBox, QComboBox {
                padding: 5px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
        """)
        
        self.n = 3
        self.k = 2
        self.mode = 1
        
        self.initUI()
        
    def initUI(self):
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #f5f5f5; border-radius: 12px;")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Заголовок
        title = QLabel("Оптимизация огневой мощи армии")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Блок параметров
        params_group = QGroupBox("Параметры задачи")
        params_layout = QVBoxLayout()
        
        # Размер матрицы
        size_box = QHBoxLayout()
        size_label = QLabel("Размер матрицы (n×n):")
        size_label.setToolTip("Количество подразделений и периодов времени")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(2, 10)
        self.size_spin.setValue(3)
        self.size_spin.valueChanged.connect(self.resize_matrix)
        size_box.addWidget(size_label)
        size_box.addWidget(self.size_spin)
        size_box.addStretch()
        params_layout.addLayout(size_box)
        
        # Коэффициент k
        k_box = QHBoxLayout()
        k_label = QLabel("Коэффициент снижения (k):")
        k_label.setToolTip("Во сколько раз снижается мощь при атаке (k > 1)")
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 10)
        self.k_spin.setValue(2)
        k_box.addWidget(k_label)
        k_box.addWidget(self.k_spin)
        k_box.addStretch()
        params_layout.addLayout(k_box)
        
        # Режим задачи
        mode_box = QHBoxLayout()
        mode_label = QLabel("Тип задачи:")
        mode_label.setToolTip("Выберите тип задачи из методички")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Задача 1 (1 атака/период, 1 атака/подразделение)", 1)
        self.mode_combo.addItem("Задачи 3-4 (2 атаки/период, 2 атаки/подразделение)", 2)
        mode_box.addWidget(mode_label)
        mode_box.addWidget(self.mode_combo)
        mode_box.addStretch()
        params_layout.addLayout(mode_box)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Матрица
        matrix_group = QGroupBox("Матрица огневой мощи")
        matrix_layout = QVBoxLayout()
        
        matrix_help = QLabel("Заполните матрицу целыми числами. C[i][j] - мощь i-го подразделения в j-й период")
        matrix_help.setWordWrap(True)
        matrix_help.setStyleSheet("color: #555; font-style: italic;")
        matrix_layout.addWidget(matrix_help)
        
        self.table = QTableWidget()
        self.table.setRowCount(3)
        self.table.setColumnCount(3)
        self.table.horizontalHeader().setDefaultSectionSize(80)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border-radius: 8px;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 5px;
                border: none;
            }
        """)
        self.init_matrix()
        matrix_layout.addWidget(self.table)
        
        matrix_group.setLayout(matrix_layout)
        layout.addWidget(matrix_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)
        
        self.calc_btn = StyledButton("Рассчитать оптимальный план")
        self.calc_btn.setToolTip("Вычислить оптимальное расписание атак")
        self.calc_btn.clicked.connect(self.calculate)
        buttons_layout.addWidget(self.calc_btn)
        
        self.clear_btn = StyledButton("Сбросить матрицу")
        self.clear_btn.setToolTip("Очистить матрицу и установить значения по умолчанию")
        self.clear_btn.setStyleSheet("background-color: #f44336;")
        self.clear_btn.clicked.connect(self.init_matrix)
        buttons_layout.addWidget(self.clear_btn)
        
        layout.addLayout(buttons_layout)
        
        # Результаты
        result_group = QGroupBox("Результаты оптимизации")
        result_layout = QVBoxLayout()
        
        self.result_text = QLabel()
        self.result_text.setWordWrap(True)
        self.result_text.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                min-height: 100px;
                color: #333;
            }
        """)
        self.result_text.setText("Здесь будут отображаться результаты расчетов")
        
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # Подвал
        footer = QLabel("Военная кафедра © 2023 | Оптимизация огневой мощи")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #777; font-size: 10px;")
        layout.addWidget(footer)
        
        # Инициализация матрицы
        self.init_matrix()
    
    def init_matrix(self):
        n = self.size_spin.value()
        self.table.setRowCount(n)
        self.table.setColumnCount(n)
        
        # Устанавливаем заголовки
        for i in range(n):
            self.table.setHorizontalHeaderItem(i, QTableWidgetItem(f"Период {i+1}"))
            self.table.setVerticalHeaderItem(i, QTableWidgetItem(f"Подр. {i+1}"))
        
        # Заполняем значениями по умолчанию
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem("1")
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont("Arial", 10))
                if i == j:
                    item.setBackground(QColor(230, 245, 255))
                self.table.setItem(i, j, item)
    
    def resize_matrix(self):
        n = self.size_spin.value()
        old_n = self.table.rowCount()
        
        self.table.setRowCount(n)
        self.table.setColumnCount(n)
        
        # Инициализация новых ячеек
        for i in range(old_n, n):
            for j in range(n):
                if j >= old_n or i >= old_n:
                    item = QTableWidgetItem("1")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(i, j, item)
    
    def get_matrix(self):
        n = self.size_spin.value()
        C = []
        for i in range(n):
            row = []
            for j in range(n):
                try:
                    val = int(self.table.item(i, j).text())
                except:
                    val = 0
                row.append(val)
            C.append(row)
        return C
    
    def calculate(self):
        try:
            C = self.get_matrix()
            k = self.k_spin.value()
            mode = self.mode_combo.currentData()
            
            if mode == 1:
                m, r = 1, 1
            else:
                m, r = 2, 2
            
            optimizer = FirepowerOptimizer(C, k, m, r)
            schedule, power = optimizer.optimize()
            
            if schedule is None:
                self.result_text.setText("Не удалось найти решение!")
                return
            
            if mode == 1:
                result_str = (f"Оптимальная перестановка: {schedule[0]}\n"
                            f"Суммарная мощь: {power:.2f}")
            else:
                result_str = (f"Расписание атак:\n"
                            f"Период | Цели\n"
                            f"----------------\n")
                for j, targets in enumerate(schedule):
                    result_str += f"{j+1:6} | {targets}\n"
                result_str += f"\nСуммарная мощь: {power:.2f}"
            
            self.result_text.setText(result_str)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при расчетах:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Настройка стиля приложения
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
    app.setPalette(palette)
    
    window = MatrixEditor()
    window.show()
    sys.exit(app.exec_())