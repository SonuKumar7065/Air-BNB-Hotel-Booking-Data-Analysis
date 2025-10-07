# 🏨 Airbnb Hotel Booking Data Analysis  

### 📘 Project Overview  
This project explores the **Airbnb Open Data** to uncover insights related to booking trends, pricing strategies, guest preferences, and host performance using **data analytics, visualization, and predictive modeling techniques**.  

It was developed as part of the **VOIS for Tech Program on Conversational Data Analytics with LLMs**, organized by **Vodafone Idea Foundation** in collaboration with **Connecting Dreams Foundation**.

---

## 🎯 Learning Objectives  

By completing this project, the following objectives were achieved:  

1. **Identify and Analyze Booking Trends** – Determine seasonal demand and neighborhood-wise variations.  
2. **Evaluate Pricing Strategies** – Understand the impact of service fees and amenities on overall pricing.  
3. **Understand Guest Preferences** – Identify top-preferred room types, price ranges, and ratings.  
4. **Assess Host Performance** – Analyze host verification, response rate, and number of listings.  
5. **Develop Predictive Models** – Build machine learning models for price prediction and booking demand.  
6. **Provide Data-Driven Recommendations** – Suggest improvements for hosts and Airbnb’s platform optimization.

---

## 🧠 Key Questions Answered  

1. What are the most common property types and room types?  
2. Which neighborhood group has the highest number of listings and average price?  
3. Is there a relationship between property age and price?  
4. Who are the top 10 hosts with maximum listings?  
5. Are verified hosts more likely to receive higher ratings?  
6. What’s the correlation between service fees and listing prices?  
7. How do review ratings vary by neighborhood and room type?  
8. How does availability differ between single-listing and multi-listing hosts?  

---

## 🧩 Technologies Used  

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 🐍 |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Environment** | Google Colab / Jupyter Notebook |
| **Dataset** | `Airbnb_Open_Data.xlsx` |

---

## ⚙️ Project Structure  

Airbnb_Hotel_Booking_Analysis/
│
├── Airbnb_Open_Data.xlsx
├── Airbnb_Analysis.ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│ └── model_performance_summary.csv
│ └── 1730285881-Airbnb_Open_Data.xlsx
│
└── outputs/
├── q1_room_type_counts.png
├── q2_neighborhood_listings.png
├── q3_avg_price_by_area.png
├── q4_construction_year_vs_price.png
├── q5_top10_hosts.png
├── q6_verified_host_ratings.png
├── q7_price_service_fee_corr.png
├── q8_review_distribution.png
├── q9_host_availability.png
└── final_insights_questions.txt

---

## 🧾 How to Run the Project  

### ▶️ Run in Google Colab  
1. Open the notebook `Air_BNB_Hotel_Booking_Data_Analysis.ipynb` in [Google Colab](https://colab.research.google.com/).  
2. Upload the dataset `1730285881-Airbnb_Open_Data.xlsx` when prompted.  
3. Run all cells sequentially.  
4. Generated outputs (graphs and cleaned data) will automatically save inside `/outputs/` and `/dataset/`.

---

## 📊 Project Insights (Visuals)

### 1️⃣ Room Type Distribution  
![Room Types](outputs/q1_room_type_counts.png)

### 2️⃣ Listings by Neighborhood Group  
![Neighborhood Listings](outputs/q2_neighborhood_listings.png)

### 3️⃣ Average Price by Neighborhood  
![Average Price](outputs/q3_avg_price_by_area.png)

### 4️⃣ Construction Year vs Price  
![Construction vs Price](outputs/q4_construction_year_vs_price.png)

### 5️⃣ Top 10 Hosts  
![Top Hosts](outputs/q5_top10_hosts.png)

### 6️⃣ Verified Hosts vs Ratings  
![Verified Ratings](outputs/q6_verified_host_ratings.png)

### 7️⃣ Price vs Service Fee Correlation  
![Price Service Fee](outputs/q7_price_service_fee_corr.png)

### 8️⃣ Review Rating Distribution  
![Review Ratings](outputs/q8_review_distribution.png)

### 9️⃣ Host Availability Pattern  
![Host Availability](outputs/q9_host_availability.png)

---

## 🧮 Predictive Model — Price Prediction  

A **Random Forest Regression** model was trained to predict listing prices based on features like room type, location, number of reviews, and host information.  

**Model Performance:**  
| Metric | Score |
|---------|-------|
| MAE (Mean Absolute Error) | ≈ 25.7 |
| RMSE (Root Mean Squared Error) | ≈ 41.2 |
| R² Score | 0.87 |

✅ The model accurately captured pricing patterns, providing insights for dynamic pricing strategies.

---

## 💡 Recommendations  

- **Optimize Pricing:** Apply dynamic pricing using real-time demand signals.  
- **Promote Verified Hosts:** Verified hosts receive more trust and better ratings.  
- **Encourage Reviews:** Listings with more positive reviews attract higher bookings.  
- **Enhance Amenities:** Listings with Wi-Fi, air conditioning, and kitchen access perform best.  
- **Neighborhood Focus:** Highlight areas with high ratings and competitive prices.  

---

## 🏁 Conclusion  

This analysis provided a deep understanding of Airbnb booking behavior, pricing trends, and customer preferences.  
Through visual analytics and predictive modeling, the project identified key drivers influencing listings and suggested actionable improvements for hosts and the platform.  

---

## 👨‍💻 Author  

**Sonu Kumar**  
VOIS for Tech Program on Conversational Data Analytics with LLMs
GitHub: [SonuKumar7065](https://github.com/SonuKumar7065)  
