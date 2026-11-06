import numpy as np


class FoodMarkovChain:
    def __init__(self):
        self.states = []  # Lưu danh sách các món ăn duy nhất (VD: ['B', 'C', 'M'])
        self.state_to_idx = {}  # Dictionary ánh xạ món ăn thành chỉ số (VD: {'B': 0, 'C': 1, 'M': 2})
        self.transition_matrix = None  # Ma trận xác suất chuyển tiếp P

    def fit(self, sequence):
        """
        Pha huấn luyện: Xây dựng Ma trận chuyển tiếp từ chuỗi lịch sử
        """
        # 1. Trích xuất các trạng thái duy nhất và sắp xếp (alphabet)
        self.states = sorted(list(set(sequence)))
        num_states = len(self.states)

        # 2. Tạo từ điển ánh xạ từ Tên món -> Index (để nhét vào ma trận)
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}

        # TODO 1: Khởi tạo ma trận đếm (Count Matrix) kích thước (num_states x num_states) toàn số 0
        # count_matrix = ...
        c_matrix = np.zeros((num_states, num_states), dtype=int)

        # TODO 2: Duyệt qua chuỗi lịch sử để đếm số lần chuyển tiếp từ trạng thái i sang j
        # Gợi ý: Dùng vòng lặp trượt từ phần tử đầu đến phần tử áp chót
        # for i in range(len(sequence) - 1):
        #     current_state = sequence[i]
        #     next_state = sequence[i+1]
        #     ... (Cộng 1 vào ô tương ứng trong count_matrix)
        for i in range(len(sequence) - 1):
            curr = sequence[i]
            next_t = sequence[i + 1]
            curr_idx = self.state_to_idx[curr]
            next_idx = self.state_to_idx[next_t]
            c_matrix[curr_idx, next_idx] += 1

        # TODO 3: Chuẩn hóa Count Matrix thành Transition Matrix (P)
        # Gợi ý: Chia từng phần tử trong một hàng cho tổng của hàng đó.
        # (Nên cộng thêm 1e-9 vào mẫu số để tránh lỗi chia cho 0 nếu có trạng thái cụt).
        # self.transition_matrix = ...
        sum_r = c_matrix.sum(axis=1, keepdims=True) + 1e-9
        self.transition_matrix = c_matrix / sum_r

    def predict_next_day(self, current_food):
        """
        Dự đoán xác suất các món ăn ngày mai dựa vào món hôm nay
        """
        if current_food not in self.state_to_idx:
            raise ValueError(f"Món '{current_food}' chưa từng xuất hiện trong lịch sử!")

        num_states = len(self.states)

        # TODO 4: Tạo vector trạng thái hiện tại (One-hot vector)
        # Gợi ý: Vector toàn số 0, chỉ vị trí của current_food là 1
        # v_current = ...
        v_curr = np.zeros(num_states)
        v_curr[self.state_to_idx[current_food]] = 1

        # TODO 5: Tính vector xác suất ngày mai
        # Gợi ý: Nhân vector v_current với self.transition_matrix bằng np.dot() hoặc @
        # v_next = ...

        v_n = v_curr @ self.transition_matrix
        # Format kết quả trả về dạng dictionary cho dễ đọc
        # return {self.states[i]: v_next[i] for i in range(num_states)}
        return {self.states[i]: v_n[i] for i in range(num_states)}


# ================= KỊCH BẢN CHẠY KIỂM THỬ =================
if __name__ == "__main__":
    # Dữ liệu lịch sử 15 ngày
    history = [
        "M",
        "B",
        "C",
        "M",
        "M",
        "B",
        "B",
        "C",
        "B",
        "M",
        "C",
        "C",
        "M",
        "B",
        "M",
    ]

    # 1. Khởi tạo và xây dựng mô hình
    model = FoodMarkovChain()
    model.fit(history)

    # 2. In Ma trận chuyển tiếp để kiểm tra
    print("--- MA TRẬN CHUYỂN TRẠNG THÁI (P) ---")
    print(f"Trạng thái (Index): {model.state_to_idx}")
    print(np.round(model.transition_matrix, 2))

    # 3. Dự đoán ngày mai nếu hôm nay ăn Mì Quảng ('M')
    today = "M"
    predictions = model.predict_next_day(today)

    print(f"\n--- DỰ ĐOÁN NGÀY MAI (Hôm nay ăn {today}) ---")
    for food, prob in predictions.items():
        print(f"Xác suất đòi ăn {food}: {prob * 100:.1f}%")