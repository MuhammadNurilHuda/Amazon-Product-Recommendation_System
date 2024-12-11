# **Amazon Products Recommendation System**

# **Project Overview**

## **Background**

E-commerce platforms such as Amazon generate vast amounts of data, including customer reviews, product ratings, and product details. With the ever-growing number of products available online, customers face challenges in finding the most relevant items to their preferences. This often leads to a phenomenon known as "choice overload," where too many options result in difficulty in decision-making and dissatisfaction.

A **recommendation system** plays a crucial role in overcoming this issue by filtering products based on user preferences and behaviors. By offering tailored suggestions, it improves the customer experience, increases engagement, and boosts sales. This project focuses on building a **content-based filtering** recommendation system for Amazon's product dataset, leveraging product metadata such as names, categories, and ratings.

## **Importance of the Project**

The importance of this project lies in its ability to address critical challenges in the e-commerce sector:

1. **Enhancing User Experience**:

   - A recommendation system ensures that users are presented with products they are most likely to engage with, reducing the time spent searching for items.

2. **Driving Sales and Revenue**:

   - By suggesting products aligned with user preferences, businesses can significantly increase their conversion rates.

3. **Handling Data Overload**:

   - With millions of products in the catalog, an intelligent system helps streamline user navigation and improves retention.

4. **Adaptability**:
   - The project builds a foundation for further extensions, such as hybrid filtering (content + collaborative) or context-aware recommendations, enhancing long-term scalability.

## **Research and References**

The project leverages state-of-the-art techniques and tools based on existing research in the recommendation systems domain. Key references include:

1. **Content-Based Filtering**:

   - This approach involves understanding the attributes of products (e.g., categories, names, descriptions) to calculate similarities between items. It has been widely used in academic and industrial applications ([Aggarwal, 2016](https://doi.org/10.1007/978-3-319-29659-3)).

2. **Natural Language Processing (NLP)**:

   - Text preprocessing techniques such as tokenization, stopword removal, and lemmatization are implemented to handle unstructured product metadata ([Manning et al., 2008](https://nlp.stanford.edu/IR-book/)).

3. **FastText Embedding**:

   - Embedding models like FastText are utilized to transform product descriptions into numerical representations that capture semantic meaning ([Bojanowski et al., 2017](https://arxiv.org/abs/1607.04606)).

4. **Applications in E-commerce**:
   - Recommendation systems have proven to increase average order value and customer satisfaction by 5–10% ([McKinsey & Company Report, 2019](https://www.mckinsey.com/)).

This project applies these principles to Amazon's product dataset to develop a robust content-based recommendation system, laying the groundwork for future integration of collaborative and hybrid filtering techniques.

# **Business Understanding**

## **Problem Statements**

The e-commerce industry, particularly platforms like Amazon, faces several challenges in ensuring a seamless and personalized shopping experience for its users. The key problems identified are:

1. **Choice Overload**:

   - With millions of products available, customers often struggle to find relevant items, leading to decision fatigue and dissatisfaction.

2. **Product Discovery**:

   - Many high-quality or niche products remain unnoticed by users due to the lack of effective recommendation mechanisms.

3. **Customer Retention**:

   - Without personalized recommendations, users may abandon the platform, resulting in reduced engagement and lower customer retention rates.

4. **Revenue Optimization**:
   - Poor recommendations can lead to missed sales opportunities, directly impacting the platform's revenue.

---

## **Goals**

The primary goal of this project is to design and implement a robust **recommendation system** that addresses the challenges outlined above. The specific objectives include:

1. **Improve User Experience**:

   - Provide customers with highly relevant product recommendations to enhance their shopping experience.

2. **Increase Product Visibility**:

   - Ensure that niche and high-quality products are discoverable through personalized recommendations.

3. **Boost Sales and Engagement**:

   - Drive higher conversion rates and repeat purchases by delivering tailored suggestions.

4. **Lay the Foundation for Advanced Systems**:
   - Develop a scalable system that can be extended to hybrid or context-aware recommendation systems in the future.

---

## **Solution Approach**

To achieve the stated goals, the project explores three solution approaches:

### **1. Content-Based Filtering**

- **Overview**:
  Content-Based Filtering relies on product metadata such as categories, descriptions, and names to compute similarities between items. Recommendations are based on the features of items that a user has previously interacted with.
- **Implementation Steps**:
  1.  Preprocess product metadata:
      - Tokenization, stopword removal, and lemmatization.
  2.  Generate embeddings for product features using FastText.
  3.  Compute a similarity matrix to identify products similar to a user's preferences.
  4.  Recommend the most similar products to the user.
- **Advantages**:
  - Handles product discovery effectively.
  - Works well for cold-start problems (new products without user interaction).
- **Challenges**:
  - Limited to the quality and depth of product metadata.
  - Cannot leverage user interaction data for recommendations.

---

### **2. Collaborative Filtering**

- **Overview**:
  Collaborative Filtering uses user-product interaction data (e.g., ratings, purchases) to identify patterns and recommend products based on user behavior.
- **Implementation Steps**:
  1.  Build a user-item interaction matrix.
  2.  Compute similarities between users or items using cosine similarity.
  3.  Recommend products based on similar users' preferences or similar items.
- **Advantages**:
  - Leverages interaction data, enabling personalized recommendations.
  - Dynamically adapts to user behavior changes.
- **Challenges**:
  - Struggles with the cold-start problem for new users or products.
  - Requires a significant amount of interaction data for optimal performance.

---

### **3. Hybrid Filtering**

- **Overview**:
  Hybrid Filtering combines the strengths of Content-Based Filtering and Collaborative Filtering to create a more robust recommendation system. It uses product metadata and user interaction data simultaneously.
- **Implementation Steps**:
  1.  Implement Content-Based Filtering to compute item similarities based on metadata.
  2.  Implement Collaborative Filtering to compute recommendations based on user-product interactions.
  3.  Combine the results of both methods using a weighted hybrid approach:
      - Assign weights to the scores from Content-Based and Collaborative Filtering (e.g., 70% Collaborative, 30% Content-Based).
  4.  Rank recommendations based on the combined scores.
- **Advantages**:
  - Addresses the cold-start problem by leveraging metadata for new products.
  - Provides personalized recommendations based on user behavior.
  - Achieves better accuracy by combining complementary methods.
- **Challenges**:
  - Higher computational complexity compared to individual methods.
  - Requires careful tuning of weights to balance both approaches effectively.

---

## **Conclusion**

The three proposed approaches—**Content-Based Filtering**, **Collaborative Filtering**, and **Hybrid Filtering**—address the challenges of choice overload, product discovery, and customer retention. Each approach has unique strengths:

- **Content-Based Filtering** ensures effective product discovery using metadata.
- **Collaborative Filtering** leverages user behavior for highly personalized recommendations.
- **Hybrid Filtering** combines the advantages of both methods, providing a balanced and robust solution.

The implementation and comparison of these methods will determine the most effective approach for the given dataset and business requirements. Hybrid Filtering, in particular, holds the potential to deliver the best overall performance by addressing the limitations of individual methods.

# **Data Understanding**

## **Dataset Information**

The dataset contains product sales data scraped from the Amazon website. The data is divided into 142 categories, each saved as a separate CSV file. Additionally, a full combined dataset is available under the name `Amazon-Products.csv`. The dataset includes detailed information about each product.

- **Number of Files**: 142 category-specific files and 1 combined dataset.
- **Total Columns per File**: Originally 10 columns, but one column (`Unnamed: 0`) is a bug.
- **Source**: The dataset is available on Kaggle. [Download Link](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/data)

### **Clarification on Columns**

- The dataset has **9 valid columns**, and 1 invalid column (`Unnamed: 0`) containing duplicate index values.
- The `Unnamed: 0` column was dropped during data preprocessing to ensure data integrity.

---

## **Feature Description**

The dataset contains the following valid features:

| **Feature Name** | **Description**                                               |
| ---------------- | ------------------------------------------------------------- |
| `name`           | The name of the product.                                      |
| `main_category`  | The main category to which the product belongs.               |
| `sub_category`   | The sub-category to which the product belongs.                |
| `image`          | The image of the product (URL or file reference).             |
| `link`           | The Amazon website reference link for the product.            |
| `ratings`        | The ratings given by Amazon customers to the product.         |
| `no_of_ratings`  | The total number of ratings provided for the product.         |
| `discount_price` | The discounted price of the product (if applicable).          |
| `actual_price`   | The original price (MRP) of the product before any discounts. |

---

## **Dataset Condition**

### **Number of Records**

- **Combined Dataset**:
  - Rows: `1,103,170` (after concatenating 142 category-specific files).
  - Columns: `9` (after dropping `Unnamed: 0`).

### **Missing Values**

- Missing values were identified in the following columns:

  - `ratings`
  - `no_of_ratings`
  - `discount_price`
  - `actual_price`

  Missing data was handled using imputation techniques such as:

  - Filling with `0` for ratings and no_of_ratings (indicating no reviews yet).
  - Median imputation for `actual_price` based on sub-category groups.

### **Duplicate Records**

- Duplicate rows were identified and removed based on:
  - `name`
  - `main_category`
  - `actual_price`

---

## **Exploratory Data Analysis (EDA)**

### **Scatter Plots**

1. **Scatter Plot: Actual Price vs Ratings**

   ![Scatter Plot actual price vs ratings](assets\scatter_plot_actual_price_vs_ratings.png)

   - **Description**: The plot shows the relationship between `actual_price` and `ratings`. The data is on a scale of 1e7.
   - **Insight**:
     - The majority of the products cluster near low prices, with ratings ranging from 0 to 5.
     - A single outlier exists with an extremely high price (> 6 million). This outlier is retained as it may contain useful insights.

2. **Scatter Plot: Actual Price vs Ratings (Excluding Outliers)**

   ![Scatter Plot actual price vs ratings (exclued outliers)](<assets\scatter_plot_actual_price_vs_ratings(exclude_outliers).png>)

   - **Description**: To better visualize the distribution of lower-priced items, the scatter plot excludes products priced above 1e6.
   - **Insight**:
     - Products with actual prices below 1 million show a clearer relationship between ratings and price. Higher-priced items tend to have more diverse ratings.

---

### **Histograms**

1. **Histogram: Ratings**

   ![Histogram of Ratings](assets\histogram_ratings.png)

- **Description**: The histogram visualizes the distribution of `ratings`.
- **Insight**:
  - A significant peak at `0` indicates a large number of unrated products or default values.
  - The majority of ratings are between `3` and `5`, skewed towards higher ratings, suggesting positive feedback from customers.

2. **Histogram: Number of Ratings**

   ![Histogram of no of Ratings](assets\histogram_no_of_ratings.png)

   - **Description**: The histogram visualizes the distribution of `no_of_ratings`.
   - **Insight**:
     - Most products have very few or zero ratings, with a significant peak near zero.
     - A small portion of products receive a high number of ratings, forming a long tail of highly popular items.

3. **Histogram: Discount Price**

   ![Histogram of discount prices](assets\histogram_discount_price.png)

   - **Description**: The histogram visualizes the distribution of `discount_price`.
   - **Insight**:
     - Most products have very low or zero discount prices.
     - A small number of products have significantly high discount prices, representing rare deals or promotional items.

4. **Histogram: Actual Price**

   ![Histogram of actual prices](assets\histogram_actual_price.png)

   - **Description**: The histogram visualizes the distribution of `actual_price`.
   - **Insight**:
     - The majority of products are priced low, with a peak near the lower end of the price range.
     - A small number of products are extremely expensive, creating a long tail in the dataset.

---

### **Bar Plots**

1. **Bar Plot: Main Category**

   ![Barplot of main category](assets\count_of_item_per_main_category.png)

   - **Description**: The bar plot shows the count of items per `main_category`.
   - **Insight**:
     - The "Accessories" category dominates the dataset, followed by "Women's Clothing" and "TV, Audio & Cameras".
     - Categories like "Music", "Pet Supplies", and "Grocery & Gourmet Foods" have the fewest items, suggesting a potential focus on lifestyle and fashion products.

2. **Bar Plot: Sub Category**

   ![Barplot of sub category](assets\count_of_item_per_sub_category.png)

   - **Description**: The bar plot shows the count of items per `sub_category`.
   - **Insight**:
     - "Watches" and "Fashion & Silver Jewellery" subcategories have the highest number of items.
     - Subcategories like "Toys Gifting Store" and "STEM Toys Store" have very few items, representing niche markets.

---

### **Pair Plot**

![Pairplot](assets\pairplot.png)

- **Description**: A pair plot visualizes the relationships between numerical features (`ratings`, `no_of_ratings`, `discount_price`, and `actual_price`) while distinguishing items by `main_category`.
- **Insight**:
  - Products from categories like "TV, Audio & Cameras" and "Appliances" tend to have higher prices and more ratings compared to other categories.
  - Discounts are distributed unevenly, with certain categories benefiting more from promotional pricing.

---

### **Word Cloud**

![Word Cloud](assets\wordcloud.png)

- **Description**: The word cloud visualizes the most frequent terms in product names.
- **Insight**:
  - Common terms include "Running", "Shoe", "Pack", "Cotton", "T-Shirt", and "Watch", indicating a strong focus on lifestyle and fashion products.
  - Specific keywords such as "Wireless" and "Bluetooth" highlight the prevalence of electronic and wearable technology products.

---

## **Violin Plot**

![Violin Plot](assets\violin_plot.png)

- **Description**: The violin plot visualizes the distribution of `ratings` across each `main_category`.
- **Insight**:
  - Most categories exhibit a concentration of high ratings (around 4–5), suggesting overall customer satisfaction.
  - Categories such as "TV, Audio & Cameras" and "Sports & Fitness" show a wider spread of ratings, indicating diverse customer opinions.
  - Some categories like "Music" and "Industrial Supplies" have smaller representation, with fewer data points.

### **Summary of Insights**

1. **Ratings**:

   - A significant number of products are unrated.
   - Most rated products receive positive feedback, with ratings clustered between 3 and 5.

2. **Customer Engagement**:

   - Only a small fraction of products receive high numbers of ratings, indicating that a few items dominate customer attention.

3. **Price Dynamics**:

   - The dataset is dominated by low-priced items, but a small number of high-priced products create a long tail.

4. **Category Representation**:

   - Categories like "Accessories" and "Women's Clothing" dominate the dataset, suggesting a potential bias toward lifestyle and fashion items.
   - Niche categories are underrepresented, presenting opportunities for specialized analysis.

5. **Product Names**:
   - Frequent keywords in product names align with lifestyle and technology trends, highlighting customer interests and market focus.

# **Data Preparation**

## **Overview**

The data preparation phase is critical to ensure that the dataset is clean, consistent, and ready for modeling. This section outlines the techniques and steps taken to clean and preprocess the dataset. Each step is necessary to address specific issues in the raw data, improve the quality of the dataset, and enhance the performance of the recommendation system.

---

## **Techniques and Steps**

### **1. Cleaning Currency Values**

- **Description**:
  - Removed currency symbols (e.g., `₹`) and commas from the `discount_price` and `actual_price` columns.
  - Converted these columns to float type for numerical operations.
- **Reason**:
  - Currency symbols and formatting are non-numerical and can cause issues during calculations or modeling.
  - Converting these columns to numerical format ensures compatibility with analysis and modeling processes.

---

### **2. Cleaning Ratings and Number of Ratings**

- **Description**:
  - Removed commas and special characters in the `ratings` and `no_of_ratings` columns.
  - Converted these columns to float type for numerical operations.
- **Reason**:
  - The presence of commas and non-numerical characters can interfere with statistical analysis and model training.
  - Ensuring consistent numerical formatting improves data quality.

---

### **3. Removing Duplicate Records**

- **Description**:
  - Removed duplicate rows based on the subset: `['name', 'main_category', 'actual_price']`.
- **Reason**:
  - Duplicate entries can distort the analysis and model predictions by introducing bias.
  - Retaining only unique records ensures the dataset accurately represents the product catalog.

---

### **4. Handling Missing Values**

- **Description**:
  - **`ratings` and `no_of_ratings`**:
    - Missing values in these columns were assumed to indicate new items that had not been rated yet. These were replaced with `0`.
  - **`discount_price` and `actual_price`**:
    - Ensured that if `actual_price` was null, `discount_price` was also null. No inconsistencies were found in this rule.
    - Missing values in `actual_price` were handled as follows:
      - If the percentage of missing values was greater than 5% for a sub-category, the entire sub-category was removed.
      - For sub-categories with less than or equal to 5% missing values, median imputation was applied.
    - Deleted rows where `actual_price` was `0`, as it is impossible for an item to have no price.
- **Reason**:
  - Missing values can introduce bias and inconsistency in the dataset, leading to inaccurate model predictions.
  - Median imputation was chosen over mean imputation as it is less sensitive to outliers, ensuring the data remains representative.

---

### **5. Handling Anomalies**

- **Description**:
  - Anomalous data was identified, such as an insect killer product with an unusually high price. Based on analysis, it was determined to be a bonus item and was removed from the dataset.
- **Reason**:
  - Anomalies can skew the results of the analysis and modeling process.
  - Removing such outliers ensures that the data reflects realistic and meaningful patterns.

---

## **Summary of Data Preparation**

The following steps were applied to clean and preprocess the dataset:

1. Cleaned currency values to ensure numerical compatibility.
2. Removed non-numerical characters and converted ratings to numeric values.
3. Deleted duplicate rows to maintain unique product records.
4. Handled missing values through a combination of deletion and imputation strategies.
5. Addressed anomalies by removing unrealistic or incorrect data points.

---

## **Importance of Data Preparation**

- **Ensures Data Quality**:
  - Clean, accurate, and consistent data is essential for reliable analysis and modeling.
- **Improves Model Performance**:
  - Well-prepared data reduces noise and bias, enhancing the predictive accuracy of recommendation systems.
- **Supports Business Goals**:
  - High-quality data enables better insights, leading to actionable recommendations that align with business objectives.

The cleaned and preprocessed dataset is now ready for the modeling phase.

# **Modeling and Results**

## **Introduction**

Three recommendation system approaches were implemented and evaluated—**Content-Based Filtering**, **Collaborative Filtering**, and **Hybrid Filtering**. Each method was tested for its ability to generate relevant recommendations, and the top-N recommendations were analyzed.

---

## **1. Content-Based Filtering**

### **Approach**

- **Metadata Used**:
  - Combined features (`main_category`, `sub_category`, and `name`) into a single column named `combined_features`.
- **Text Normalization**:
  1. Remove stopwords.
  2. Convert text to lowercase.
  3. Apply lemmatization.
- **Vectorization**:
  - Used FastText embeddings to convert text data into numerical vectors.
- **Dataset**:
  - Randomly reduced to 10,000 rows due to memory constraints (original dataset required ~500 GB for cosine similarity).

### **Results**

- **Test Case**:

  - Product Name: `AB Beauty House Gold Plated Bracelet Bangles With RED Set for Women and Girls 2pcs 2.8`.
  - Product Index: 250.

  ![Content-Based results](assets\content-based_result.png)

- **Outcome**:
  - Highly relevant recommendations, as shown in the visualization.
  - **Performance Metrics**:
    - **Precision**: 100%.
    - **Recall**: 83%.

### **Strengths**:

- Highly effective for datasets with rich item metadata.
- Excellent in addressing the cold start problem for new products.

### **Weaknesses**:

- Computationally expensive for large datasets.
- Dependent on metadata quality.

---

## **2. Collaborative Filtering**

### **Approach**

- **Text Normalization**:
  1. Remove stopwords.
  2. Convert text to lowercase.
  3. Remove punctuations.
- **Dataset**:
  - Same reduced dataset (10,000 rows) as used in Content-Based Filtering.
- **Simulated User Data**:
  - Proxy users were simulated due to the absence of actual user interaction data.

### **Results**

- **Test Case**:

  - Product Name: `ab beauty house gold plated bracelet bangles red set women girls 2pcs 28`.

  ![Collaborative results](assets\collaborative_result.png)

- **Outcome**:
  - Recommendations were less relevant compared to Content-Based Filtering, as shown in the visualization.
  - Limitations arose from the lack of actual user data and the cold start problem.

### **Strengths**:

- Conceptually simpler as it relies on user interaction patterns instead of item features.
- Can provide highly personalized recommendations with real user data.

### **Weaknesses**:

- Struggles with sparsity and cold start problems without sufficient user data.
- Simulated user data limited its performance.

---

## **3. Hybrid Filtering**

### **Approach**

- **Combination of Methods**:
  - Weighted combination of Content-Based (weight = 0.8) and Collaborative Filtering (weight = 0.2).
- **Dataset**:
  - Same reduced dataset (10,000 rows).

### **Results**

![Hybrid results](assets\hybrid_result.png)

- **Outcome**:
  - Results were less relevant due to the limitations of simulated user data in the Collaborative Filtering component.
  - However, the approach demonstrated the potential to balance the strengths of both methods.

### **Strengths**:

- Combines advantages of Content-Based and Collaborative Filtering.
- Handles diverse scenarios and adapts to dynamic user preferences.

### **Weaknesses**:

- Performance depends on the quality of the Collaborative Filtering component.
- Requires careful weight tuning.

---

## **Comparison of Approaches**

| **Method**              | **Strengths**                                                                               | **Weaknesses**                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Content-Based Filtering | Highly relevant recommendations; effective in cold-start scenarios.                         | Requires high-quality metadata; computationally expensive for large datasets.                 |
| Collaborative Filtering | Simpler; leverages interaction patterns for personalized recommendations.                   | Requires real user data; struggles with cold start and sparsity issues.                       |
| Hybrid Filtering        | Combines strengths of both methods; handles diverse scenarios and dynamic user preferences. | Performance depends on weight tuning; negatively impacted by poor-quality Collaborative data. |

---

# **Evaluation**

## **Evaluation Metrics**

To evaluate the performance of the recommendation systems, **Precision** and **Recall** were used:

### **1. Precision**

- **Formula**:

  ![Precision](assets\precision.png)

- **Description**:
  - Precision measures the proportion of recommended items that are relevant to the user.
  - A high precision indicates that the system delivers relevant recommendations.

### **2. Recall**

- **Formula**:

  ![Recall](assets\recall.png)

- **Description**:
  - Recall measures the proportion of relevant items that the system successfully recommends.
  - A high recall indicates that the system does not miss many relevant items.

---

## **Results**

| **Method**              | **Precision** | **Recall** |
| ----------------------- | ------------- | ---------- |
| Content-Based Filtering | 100%          | 83%        |
| Collaborative Filtering | unmeasured    | unmeasured |
| Hybrid Filtering        | 100%          | 100%       |

The 100% result in the hybrid approach is achieved because the same preferences were used as the output, creating the impression that all the recommended results are correct.

### **Analysis**

1. **Content-Based Filtering**:

   - Achieved the highest precision and recall, demonstrating its ability to provide accurate and relevant recommendations.
   - Relies heavily on metadata, which was high-quality and abundant in this dataset.

2. **Collaborative Filtering**:

   - Performed poorly due to the lack of actual user data, leading to suboptimal recommendations.
   - This highlights the importance of real user interaction data for this approach.

3. **Hybrid Filtering**:
   - Improved performance compared to Collaborative Filtering but fell short of Content-Based Filtering due to reliance on simulated user data.
   - The weighted combination allows for balancing the strengths of both methods but requires further refinement.

---

## **Conclusion**

1. **Content-Based Filtering** is the most effective method for the given dataset, offering high precision and recall due to its reliance on item features. This approach may be particularly suitable for marketplaces where users are often searching for items that are similar to the ones they are currently viewing or have purchased previously.
2. **Collaborative Filtering** is limited by the absence of real user data, making it less effective in this context. However, this approach is highly effective for use cases such as music streaming platforms (e.g., YouTube or Spotify), where users typically don't search for the exact same item but rather explore related content based on interaction patterns, such as different songs by the same artist or within the same genre.

3. **Hybrid Filtering** has potential but requires better Collaborative Filtering performance to fully leverage its strengths. This approach is versatile and well-suited for a wide range of applications because it combines item similarity (Content-Based) with user interaction data (Collaborative). This dual consideration allows hybrid systems to generate recommendations that are both diverse and aligned with user preferences.

For future implementations:

- Incorporating real user interaction data could significantly enhance the performance of Collaborative Filtering by uncovering authentic user-product relationships.
- Hybrid Filtering should be revisited with improved Collaborative Filtering components to achieve more balanced and robust recommendations, ensuring applicability across diverse use cases.
