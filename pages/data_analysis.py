import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import warnings
import sys
import streamlit as st
import time
import os

# Bỏ qua cảnh báo nếu không có tùy chọn cảnh báo hệ thống
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42) # Đặt seed ngẫu nhiên để đảm bảo kết quả nhất quán

def data_analysis():
    # Tiêu đề ứng dụng
    st.title("Phân khúc và Phân tích Khách hàng")

    st.subheader("Tải và Chuẩn bị Tập dữ liệu")

    # Tải tập dữ liệu
    dir_path = os.path.abspath(os.curdir) # Lấy đường dẫn thư mục hiện tại

    file_path = os.path.join(dir_path, "data/marketing_campaign.csv")  # Đường dẫn tệp trong dự án

    # Đọc tệp
    data = pd.read_csv(file_path, sep = ",")

    st.image('./images/Dataset explanation.png')  # Hiển thị hình ảnh từ thư mục 'images'

    # Thêm một expander để hiển thị dữ liệu thô
    with st.expander("Hiển thị Dữ liệu Thô"):
        st.write(data)

    # Xóa các giá trị bị thiếu
    data = data.dropna()

    # Kỹ thuật đặc trưng (feature engineering - như đã đề cập trước đó)
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format='mixed')
    dates = [i.date() for i in data["Dt_Customer"]]

    # Kỹ thuật Đặc trưng: Customer_For (Thời gian khách hàng là thành viên)
    d1 = max(dates)
    days = [(d1 - i).days for i in dates]
    data["Customer_For"] = days
    data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

    # Kỹ thuật Đặc trưng: Age (Tuổi)
    data["Age"] = 2025 - data["Year_Birth"]

    # Tổng chi tiêu cho các mặt hàng khác nhau
    data["Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]

    # Kỹ thuật đặc trưng bổ sung
    data["Living_With"] = data["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone"})
    data["Children"] = data["Kidhome"] + data["Teenhome"]
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
    data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

    # Đổi tên cột cho rõ ràng
    data = data.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})

    # Bỏ các đặc trưng dư thừa
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop(to_drop, axis=1)
    # Bỏ các giá trị ngoại lai cho Tuổi và Thu nhập
    data = data[(data["Age"] < 90)]
    data = data[(data["Income"] < 600000)]

    # Thêm một expander để hiển thị dữ liệu đã làm sạch với các đặc trưng mới được tạo
    with st.expander(f"Hiển thị Dữ liệu đã làm sạch ({len(data)} bản ghi) và các đặc trưng bổ sung"):
        st.write(data)
        # Hiển thị các cột liên quan bao gồm các đặc trưng đã được tạo
        st.write(data[['Recency', 'Age', 'Spent', 'Is_Parent', 'Living_With', 'Children', 'Family_Size']])

    # Biểu đồ cặp (Pairplot) của các đặc trưng được chọn
    st.header("""Vẽ biểu đồ một số đặc trưng được chọn""")
    To_Plot = ["Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
    fig = sns.pairplot(data[To_Plot], hue="Is_Parent", palette=["#682F2F", "#F3AB60"])
    st.subheader("Biểu đồ Cặp (Pairplot) của các Đặc trưng được chọn")
    st.pyplot(fig)
    st.write("""
    ### Giải thích Biểu đồ Cặp (Pairplot):
    Biểu đồ cặp trực quan hóa mối quan hệ giữa các đặc trưng được chọn trong tập dữ liệu. Các biểu đồ trên đường chéo hiển thị phân phối của từng đặc trưng, trong khi các biểu đồ ngoài đường chéo hiển thị mối quan hệ cặp đôi giữa các đặc trưng.

    **Các Đặc trưng Chính**:
    - **Income (Thu nhập)**: Đại diện cho thu nhập của khách hàng.
    - **Recency (Gần đây)**: Số ngày kể từ lần mua hàng cuối cùng của khách hàng.
    - **Customer_For (Thời gian là khách hàng)**: Số ngày kể từ khi khách hàng đăng ký.
    - **Age (Tuổi)**: Tuổi của khách hàng.
    - **Spent (Chi tiêu)**: Tổng số tiền khách hàng đã chi tiêu.
    - **Is_Parent (Là Phụ huynh)**: Cho biết khách hàng có phải là phụ huynh (1) hay không (0).

    **Màu sắc (Hue - Is_Parent)**:
    - Tham số **hue** được đặt thành `Is_Parent`, do đó khách hàng là phụ huynh được vẽ bằng một màu (ví dụ: `#682F2F`), và không phải phụ huynh bằng một màu khác (ví dụ: `#F3AB60`).
    - Điều này giúp bạn phân biệt trực quan giữa hai nhóm và phân tích cách các biến khác nhau giữa chúng.
    """)

    # Ma trận tương quan
    st.subheader("Ma trận Tương quan")
    # Loại trừ các cột không phải số trước khi tính toán ma trận tương quan
    numeric_data = data.select_dtypes(include=[np.number])
    # Tính ma trận tương quan chỉ trên các cột số
    corrmat = numeric_data.corr()
    fig_corr = plt.figure(figsize=(20,20))
    sns.heatmap(corrmat, annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig_corr)
    st.write("""
    ### Giải thích Ma trận Tương quan:
    Ma trận tương quan hiển thị mối quan hệ giữa các đặc trưng số trong tập dữ liệu.
    Các giá trị nằm trong khoảng từ -1 đến 1:
    
    - **Tương quan dương** (gần 1): Cho thấy khi một đặc trưng tăng, đặc trưng kia cũng tăng.
    - **Tương quan âm** (gần -1): Cho thấy khi một đặc trưng tăng, đặc trưng kia giảm.
    - **Không tương quan** (gần 0): Cho thấy không có mối quan hệ có ý nghĩa giữa các đặc trưng.

    **Những Thông tin Chính**:
    - **Thu nhập (Income) và Chi tiêu (Spent)**: Tương quan dương có thể gợi ý rằng những cá nhân có thu nhập cao hơn có xu hướng chi tiêu nhiều hơn.
    - **Tuổi (Age) và Thời gian là khách hàng (Customer_For)**: Tương quan âm, vì những khách hàng lớn tuổi hơn có thể đã đăng ký trong thời gian dài hơn.
    - **Là Phụ huynh (Is_Parent) và Chi tiêu (Spent)**: Bạn có thể thấy sự khác biệt về chi tiêu giữa phụ huynh và không phải phụ huynh.
    """)

    st.header("""Tiền xử lý dữ liệu""")
    # Lấy danh sách các biến phân loại
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)

    # Mã hóa nhãn (Label Encoding) cho các kiểu dữ liệu đối tượng (object dtypes).
    LE = LabelEncoder()
    for i in object_cols:
        data[i] = data[[i]].apply(LE.fit_transform)

    # Tạo một bản sao của dữ liệu
    ds = data.copy()
    # Tạo một tập con của dataframe bằng cách loại bỏ một số đặc trưng nhất định
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
    ds = ds.drop(cols_del, axis=1)
    
    # Chuẩn hóa (Scaling)
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)

    # Thêm một expander để hiển thị dataframe sẽ được sử dụng cho việc mô hình hóa tiếp theo
    with st.expander("Dataframe sẽ được sử dụng cho việc mô hình hóa tiếp theo"):
        st.write(scaled_ds.head())

    st.header("""Giảm chiều dữ liệu""")
    # Khởi tạo PCA để giảm chiều dữ liệu
    pca = PCA(n_components=3)
    pca.fit(scaled_ds)
    PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=["col1", "col2", "col3"])

    # Phép chiếu 3D của dữ liệu trong không gian giảm chiều
    x = PCA_ds["col1"]
    y = PCA_ds["col2"]
    z = PCA_ds["col3"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="maroon", marker="o")
    ax.set_title("Phép chiếu 3D của Dữ liệu trong Không gian giảm chiều")
    st.pyplot(fig)
    st.write("""
    ### Giải thích Phép chiếu 3D:
    Biểu đồ phân tán 3D này trực quan hóa tập dữ liệu sau khi áp dụng PCA để giảm chiều thành ba thành phần (`col1`, `col2`, `col3`).
    Mỗi điểm trong biểu đồ đại diện cho một khách hàng trong không gian 3D đã được giảm chiều.
    """)

    st.header("""Phân cụm""")
    # Kiểm tra nhanh phương pháp Elbow để tìm số lượng cụm
    st.write('Phương pháp Elbow để xác định số lượng cụm sẽ được hình thành:')

    # Phương pháp Elbow thủ công với thời gian phù hợp (fit time)
    X = PCA_ds.values  # Lấy dữ liệu dưới dạng ma trận
    distortions = []
    fit_times = []
    K_range = range(1, 11)

    for k in K_range:
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') # n_init='auto' để tránh cảnh báo
        kmeans.fit(X)
        end_time = time.time()

        distortions.append(kmeans.inertia_)
        fit_times.append(end_time - start_time)

    # Tạo biểu đồ
    fig_elbow, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(K_range, distortions, marker='o', color='tab:blue')
    ax1.set_xlabel('Số lượng cụm')
    ax1.set_ylabel('Inertia', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # Khởi tạo trục y thứ hai cho thời gian phù hợp
    ax2.plot(K_range, fit_times, marker='o', color='tab:red')
    ax2.set_ylabel('Thời gian phù hợp (giây)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Thêm một đường thẳng đứng tại k=4 (hoặc bất kỳ điểm 'khuỷu tay' nào)
    ax1.axvline(x=4, color='gray', linestyle='--')

    ax1.set_title('Phương pháp Elbow để tìm k tối ưu với thời gian phù hợp')

    st.pyplot(fig_elbow)
    st.write("""
    ### Giải thích Phương pháp Elbow với thời gian phù hợp:
    - **Đường màu xanh** hiển thị **quán tính (inertia)** (tổng bình phương khoảng cách trong cụm).
    - **Đường màu đỏ** hiển thị **thời gian phù hợp (fit time)** (bằng giây) cho mỗi giá trị của `k`.
    - **Đường màu xám thẳng đứng** tại `k=4` cho biết số lượng cụm tối ưu đã chọn.
    """)

    # Khởi tạo mô hình Phân cụm Phân cấp (Agglomerative Clustering)
    AC = AgglomerativeClustering(n_clusters=4)
    yhat_AC = AC.fit_predict(PCA_ds)
    PCA_ds["Clusters"] = yhat_AC
    data["Clusters"] = yhat_AC

    # Vẽ biểu đồ các cụm
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
    ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
    ax.set_title("Biểu đồ các Cụm")
    st.pyplot(fig)
    st.write("""
    ### Giải thích Biểu đồ Cụm:
    Biểu đồ phân tán 3D này hiển thị kết quả phân cụm của Phân cụm Phân cấp (Agglomerative Clustering) với 4 cụm.
    Mỗi điểm được tô màu dựa trên cụm được gán, và biểu đồ cung cấp thông tin chi tiết về sự phân bố của các cụm trong không gian 3D đã giảm chiều.
    """)

    st.header("""Đánh giá mô hình""")

    # Biểu đồ Phân phối Cụm
    st.subheader("Phân phối Cụm")
    pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
    fig, ax = plt.subplots(figsize=(8, 6))  # Tạo một subplot 2D
    pl = sns.countplot(x=data["Clusters"], palette=pal, ax=ax)  # Sử dụng countplot đúng cách
    pl.set_title("Phân phối các Cụm")
    st.pyplot(fig)  # Hiển thị biểu đồ trong Streamlit
    st.write("""
    ### Giải thích Phân phối Cụm:
    Biểu đồ đếm (count plot) hiển thị cách dữ liệu được phân bố trên các cụm khác nhau.
    Mỗi thanh đại diện cho số lượng điểm dữ liệu (khách hàng) trong một cụm cụ thể.
    Điều này cho phép bạn thấy có bao nhiêu khách hàng thuộc về mỗi cụm sau khi áp dụng Phân cụm Phân cấp (Agglomerative Clustering).
    """)

    # Hồ sơ Cụm dựa trên Thu nhập và Chi tiêu
    st.subheader("Hồ sơ Cụm dựa trên Thu nhập và Chi tiêu")
    fig, ax = plt.subplots(figsize=(8, 6))  # Tạo một subplot 2D
    pl_1 = sns.scatterplot(data=data, x="Spent", y="Income", hue="Clusters", palette=pal, ax=ax)
    pl_1.set_title("Hồ sơ Cụm dựa trên Thu nhập và Chi tiêu")
    plt.legend()  # Thêm chú giải vào biểu đồ phân tán
    st.pyplot(fig)  # Hiển thị biểu đồ trong Streamlit
    st.write("""
    ### Giải thích Hồ sơ Cụm:
    Biểu đồ phân tán này giúp trực quan hóa cách khách hàng trong các cụm khác nhau được phân bố dựa trên chi tiêu (Spent) và thu nhập (Income) của họ.
    Mỗi điểm đại diện cho một khách hàng, và màu sắc tương ứng với cụm mà họ thuộc về.
    """)

    # Swarmplot và Boxplot cho Phân phối Chi tiêu (Spent) trên các Cụm
    st.subheader("Phân phối Chi tiêu trên các Cụm")
    fig, ax = plt.subplots(figsize=(8, 6))  # Tạo một subplot 2D
    # Không cần dòng plt.figure(figsize=(8, 6)) này vì đã có fig, ax = plt.subplots()
    pl_2 = sns.swarmplot(x=data["Clusters"], y=data["Spent"], color="#CBEDDD", alpha=0.5, ax=ax)
    pl_3 = sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal, ax=ax)
    # plt.show()  # Không cần dòng này trong Streamlit, thay bằng st.pyplot()
    st.pyplot(fig)  # Hiển thị biểu đồ trong Streamlit
    st.write("""
    ### Giải thích Phân phối Chi tiêu:
    Sự kết hợp giữa swarmplot và boxenplot cho thấy sự phân bố chi tiêu của khách hàng (Spent) trên các cụm.
    - **Swarmplot** hiển thị các điểm dữ liệu riêng lẻ.
    - **Boxenplot** hiển thị sự phân bố, bao gồm trung vị và các khoảng tứ phân vị.
    Điều này giúp so sánh cách chi tiêu khác nhau giữa các cụm.
    """)

    # Biểu đồ đếm (Countplot) cho Tổng số Khuyến mãi được Chấp nhận dựa trên Cụm
    st.subheader("Tổng số Khuyến mãi được Chấp nhận theo Cụm")
    fig, ax = plt.subplots(figsize=(8, 6))  # Tạo một subplot 2D
    data["Total_Promos"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"]
    pl_4 = sns.countplot(x=data["Total_Promos"], hue=data["Clusters"], palette=pal, ax=ax)
    pl_4.set_title("Số lượng Khuyến mãi đã Chấp nhận")
    pl_4.set_xlabel("Tổng số Khuyến mãi đã Chấp nhận")
    st.pyplot(fig)  # Hiển thị biểu đồ trong Streamlit
    st.write("""
    ### Giải thích Chấp nhận Khuyến mãi:
    Biểu đồ đếm này hiển thị sự phân bố số lượng khuyến mãi được khách hàng chấp nhận trên các cụm khác nhau.
    Mỗi thanh đại diện cho số lượng khách hàng đã chấp nhận một số lượng khuyến mãi cụ thể, và màu sắc cho biết cụm.
    """)

    # Boxenplot cho Số lượng Giao dịch đã Mua theo Cụm
    st.subheader("Số lượng Giao dịch đã Mua trên các Cụm")
    fig, ax = plt.subplots(figsize=(8, 6))  # Tạo một subplot 2D
    pl_5 = sns.boxenplot(y=data["NumDealsPurchases"], x=data["Clusters"], palette=pal, ax=ax)
    pl_5.set_title("Số lượng Giao dịch đã Mua")
    st.pyplot(fig)  # Hiển thị biểu đồ trong Streamlit
    st.write("""
    ### Giải thích Giao dịch đã Mua:
    Biểu đồ boxenplot này hiển thị sự phân bố số lượng giao dịch được khách hàng mua trong mỗi cụm.
    Nó cung cấp thông tin chi tiết về cách mỗi cụm phản ứng với các giao dịch khuyến mãi khác nhau.
    """)

    st.header("""Hồ sơ khách hàng (Profiling)""")

    # Các đặc trưng cá nhân để lập hồ sơ
    Personal = ["Kidhome", "Teenhome", "Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education", "Living_With"]

    # Lặp qua từng đặc trưng cá nhân và tạo biểu đồ jointplot cho chi tiêu
    for i in Personal:
        st.subheader(f"Jointplot: {i} so với Chi tiêu")
        # Tạo biểu đồ jointplot cho từng đặc trưng cá nhân so với Chi tiêu, được nhóm theo cụm
        fig, ax = plt.subplots(figsize=(8, 6)) # Tạo một figure mới cho mỗi jointplot
        jointplot = sns.jointplot(x=data[i], y=data["Spent"], hue=data["Clusters"], kind="kde", palette=pal)
        st.pyplot(jointplot.fig)  # Hiển thị biểu đồ trong Streamlit

        # Mô tả từng biểu đồ
        st.write(f"""
        ### Giải thích Biểu đồ cho {i}:
        Biểu đồ jointplot này cho thấy mối quan hệ giữa đặc trưng `{i}` và tổng số tiền khách hàng đã chi tiêu.
        Biểu đồ sử dụng phương pháp **Ước lượng Mật độ Hạt nhân (KDE)** để trực quan hóa sự phân bố chi tiêu cho mỗi cụm.
        **Màu sắc (hue)** được đặt thành `Clusters` để trực quan hóa cách khách hàng từ các cụm khác nhau được phân bố dựa trên đặc trưng này và hành vi chi tiêu của họ.
        
        - **Trục X**: Đại diện cho các giá trị của đặc trưng `{i}`.
        - **Trục Y**: Đại diện cho số tiền khách hàng đã chi tiêu (`Spent`).
        - Các **đường cong KDE** chỉ ra mật độ chi tiêu trên phạm vi của đặc trưng `{i}`.
        
        Trực quan hóa này giúp hiểu cách các cụm khác nhau hành xử liên quan đến đặc trưng này và cách nó ảnh hưởng đến chi tiêu.
        """)
    st.image('images/Dataset insights.png')  # Hiển thị hình ảnh từ thư mục 'images'