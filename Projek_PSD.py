import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy.stats import pearsonr, spearmanr

# Load dataset
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv', sep=';')

# Preprocess data
fitur = df.drop(columns=['target'])
target = df['target']

# Korelasi
pearson_corr = {}
spearman_corr = {}
for column in fitur.columns:
    corr_pearson, _ = pearsonr(fitur[column], target)
    corr_spearman, _ = spearmanr(fitur[column], target)
    pearson_corr[column] = corr_pearson
    spearman_corr[column] = corr_spearman

correlation_df = pd.DataFrame({
    'Feature': fitur.columns,
    'Pearson Correlation': list(pearson_corr.values()),
    'Spearman Correlation': list(spearman_corr.values())
})

# Streamlit interface with sidebar
with st.sidebar:
    page = option_menu("Menu", ["Data", "Analisis Data", "Prediksi"])
    split_ratio = st.selectbox("Pilih Rasio Pembagian Data:", ["90:10", "70:30", "80:20"])

# Determine test size based on selected split ratio
if split_ratio == "90:10":
    test_size = 0.1
elif split_ratio == "80:20":
    test_size = 0.2
else:
    test_size = 0.3

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Train models
models = {
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
}
accuracies = {}
confusion_matrices = {}
classification_reports = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)
    confusion_matrices[name] = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=[f'Predicted {i}' for i in range(len(np.unique(y)))],
        index=[f'Actual {i}' for i in range(len(np.unique(y)))]
    )
    classification_reports[name] = classification_report(y_test, y_pred, output_dict=True)

# KNN Validation and Test Accuracies
k_values = range(1, 11)
validation_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Cross-validation untuk mendapatkan akurasi validasi
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    validation_accuracies.append(cv_scores.mean())
    
    # Fit model dan hitung akurasi pada test set
    knn.fit(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    test_accuracies.append(test_accuracy)

if page == "Analisis Data":
    st.title("Analisis Data Kesehatan Jantung")

    # Tampilkan tabel korelasi
    st.header("Korelasi Fitur")
    st.dataframe(correlation_df)

    # Tampilkan akurasi model
    st.header("Akurasi Model")
    st.write("Akurasi masing-masing model:")
    for model, accuracy in accuracies.items():
        st.write(f"{model}: {accuracy:.2f}")

    # Tampilkan recall, precision, dan F1-score
    st.header("Hasil recall, presisi, dan F1-Score")
    for model, report in classification_reports.items():
        st.write(f"**{model}**")
        st.write("Precision: ", report['1']['precision'])
        st.write("Recall: ", report['1']['recall'])
        st.write("F1 Score: ", report ['1']['f1-score'])
        st.write("---")

    # Tampilkan diagram batang untuk perbandingan akurasi
    st.header("Perbandingan Akurasi Model")
    st.bar_chart(pd.Series(accuracies))

    # Tampilkan matriks kebingungan
    st.header("Confussion Matrics")
    selected_model = st.selectbox("Pilih model untuk melihat Confusion Matrics:", list(models.keys()))
    st.write(f"Confusion Matrics for {selected_model}:")
    st.dataframe(confusion_matrices[selected_model])

    # Validasi dan pengujian KNN
    st.header("Validasi dan Pengujian KNN")
    st.write("Nilai K dan Akurasi:")
    knn_data = pd.DataFrame({
        'K': k_values,
        'Validation Accuracy': validation_accuracies,
        'Test Accuracy': test_accuracies
    })
    st.dataframe(knn_data)

    # Tampilkan grafik KNN
    st.header("Grafik Akurasi KNN")
    knn_chart_data = pd.DataFrame({
        'K': k_values,
        'Validation Accuracy': validation_accuracies,
        'Test Accuracy': test_accuracies
    })
    st.line_chart(knn_chart_data.set_index('K'))

elif page == "Prediksi":
    st.title("Prediksi dengan Machine Learning")

    # Prediksi interaktif
    st.header("Masukkan Data untuk Prediksi")
    input_data = []
    for col in X.columns:
        value = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)
        input_data.append(value)

    selected_prediction_model = st.selectbox("Pilih model untuk prediksi:", list(models.keys()))
    if st.button("Prediksi"):
        model = models[selected_prediction_model]
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        st.success(f"Hasil prediksi: {'Penyakit Terdeteksi' if prediction == 1 else 'Tidak Terdeteksi'}")

elif page == "Data":
    st.title("Data yang Digunakan")

    # Tampilkan dataset
    st.header("Dataset Kesehatan Jantung")
    st.write("Dataset ini terdiri dari 1190 instance dengan 11 fitur. Kumpulan data ini dikumpulkan dan digabungkan di satu tempat untuk membantu memajukan penelitian tentang pembelajaran mesin terkait CAD dan algoritma penambangan data, dan diharapkan pada akhirnya dapat memajukan diagnosis klinis dan pengobatan dini. Dataset ini bersal dari Cleveland negara Hungaria")
    st.dataframe(df)