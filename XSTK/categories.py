import numpy as np


class RecruitmentCategoricalNB:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.likelihoods = {}
        self.vocab_sizes = {}

    def fit(self, X, y):
        N, num_features = X.shape
        self.classes = np.unique(y)

        # Đếm số lượng giá trị phân biệt (K_i) cho từng đặc trưng
        for i in range(num_features):
            self.vocab_sizes[i] = len(np.unique(X[:, i]))
            self.likelihoods[i] = {c: {} for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]
            N_c = X_c.shape[0]
            self.priors[c] = N_c / N

            for i in range(num_features):
                K_i = self.vocab_sizes[i]
                unique_values = np.unique(X[:, i])

                for v in unique_values:
                    # TODO 1: Tính likelihood cho giá trị v của đặc trưng i trong lớp c
                    # count_v_in_c = ...
                    # self.likelihoods[i][c][v] =
                    Ntic = np.sum(X_c[:, i] == v)
                    self.likelihoods[i][c][v] = (Ntic + 1) / (N_c + K_i) # alpha cho = 1
                    
    def predict_score(self, x):
        scores = {}
        for c in self.classes:
            # TODO 2: Tính điểm số (Score) bằng cách nhân Prior với Likelihood của 4 đặc trưng đầu vào
            # score = ...
            score = self.priors[c]
            for i,v in enumerate(x):
                score *= self.likelihoods[i][c][v]
            scores[c] = score
        return scores


# ================= DỮ LIỆU HUẤN LUYỆN =================
X_train = np.array(
    [
        ["IT", "Nam_1", "Tot", "Co"],
        ["IT", "Nam_2", "Trung_binh", "Co"],
        ["Co_Dien_Tu", "Nam_1", "Tot", "Khong"],
        ["Khac", "Nam_3", "Trung_binh", "Khong"],
        ["IT", "Nam_1", "Trung_binh", "Khong"],
        ["Co_Dien_Tu", "Nam_2", "Tot", "Co"],
    ]
)
y_train = np.array(["Pass", "Pass", "Pass", "Fail", "Fail", "Pass"])

# Ứng viên mới: Sinh viên Cơ Điện Tử, Năm 1, Toán Trung bình, Có biết Python
X_test = np.array(["Co_Dien_Tu", "Nam_1", "Trung_binh", "Co"])

model = RecruitmentCategoricalNB()
model.fit(X_train, y_train)
y_pred = model.predict_score(X_test)

# Có cho ứng viên pass không?

print("Điểm số các lớp:", y_pred)

du_doan = max(y_pred, key=y_pred.get)
print("Kết quả dự đoán:", du_doan)