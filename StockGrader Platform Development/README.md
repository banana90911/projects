The Pennsylvania State University <br>
Data Science, B.S. (Engineering) <br>
Senior Capstone Project <br>

# StockGrader.io Final Report

Siheon Jung, Adith Gopal, Hongyu Guo <br>
spj5294@psu.edu, azg5941@psu.edu, hqg5209@psu.edu 




ABSTRACT
In the contemporary digital financial era, the challenge of making stock market investing approachable and comprehensible for novice investors, particularly those belonging to Generation Z, has never been more pressing. Amidst this backdrop, StockGrader.io emerges as a unique platform, designed to improve the investing process by employing advanced predictive analytics centered around utilizing a Long Short-Term Memory (LSTM) model. This state-of-the-art approach enables the platform to accurately forecast stock prices for stocks within the S&P 500 index. By utilizing LSTM's unparalleled ability to analyze sequential time-series data with real-time market insights, StockGrader.io aims to dispel the complexities surrounding stock investment. Additionally, unique composite, single-value Stock Grades are created to represent the current performance of a stock, using multiple market indicators and weighing them accordingly. This simplifies decision-making for those new to the financial markets and significantly contributes to enhancing financial literacy. These algorithms are developed utilizing Python and Yahoo! Finance’s free API on the backend in combination with a Streamlit-based frontend application. The forthcoming report explains the development journey of StockGrader.io, spotlighting the strategic selection of LSTM for its predictive prowess in stock price movements. It further articulates the application’s intended impact on empowering emerging investors by improving their understanding and strategic investment guidance. Some constraints in the development of this tool include financial budget and time. Future work can provide additional resources to StockGrader.io to improve its impact and range of audience, including incorporating additional stock indexes and developing a StockGrader.io mobile application. 


Table of Contents


Abstract 											          1
Table of Contents 										          2
Introduction 											          3
Literature Review 										          6 
Methodology 										          7
Planning
Front-end
Back-end (LSTM)
Back-end (Stock Grading)
Risks
Logical Testing
Back-end Unit Testing
Front-end Unit Testing
Conclusions										        15
Pros
Cons
Challenges
Lessons Learned
Future Work									  	        17
References											        18

1.  Introduction
As digital technology reshapes the financial sector, a significant gap persists in accessing investment tools for experienced versus novice investors. StockGrader.io emerges as a solution designed to empower the latter group, emphasizing Generation Z investors. Despite their keen interest in the stock market, these individuals often find themselves overwhelmed by the sheer amount of information publicly available. StockGrader.io addresses this by providing advanced analytics with LSTM and our unique Stock Grade, but also provides descriptions in layman’s terms of the algorithms' significance, enhancing a user’s understanding of the S&P 500 stock index.
The complexity of the stock market, characterized by volatile price movements and an overwhelming amount of data, presents a tough challenge for novice investors. The development of StockGrader.io, is motivated by the need to provide an effective yet digestible tool for stock price prediction and evaluation. LSTM models are renowned for their ability to learn from data sequences, making them exceptionally suited for the time-series data inherent in stock price movements. This capability allows for more accurate and reliable predictions, a critical component in the price prediction analysis within StockGrader.io.
The core purpose of StockGrader.io is to democratize stock market investing by making it accessible and comprehensible to a broader audience. The integration of LSTM into StockGrader.io serves several key objectives: to enhance the platform's predictive accuracy, to provide users with actionable insights based on sophisticated data analysis, and to educate users on the potential future performance of stocks. By doing so, StockGrader.io aims to remove barriers to investment for individuals lacking in financial expertise, ultimately fostering a more inclusive investing culture.
The incorporation of LSTM models into StockGrader.io represents a significant technical advancement in the prediction of stock prices. LSTMs are particularly appropriate for this application due to their effectiveness in handling financial market volatility and temporal dependencies. They can capture long-term dependencies and patterns in historical stock data, which traditional models might overlook. This predictive capability is crucial for providing StockGrader.io users with reliable stock grades, which are directly influenced by the projected future performance of stocks. Accordingly, our application will use LSTM for stock price prediction, but we have designed a CNN-LSTM model that could be incorporated in future work, as there is potential for a CNN-LSTM model to perform better in stock price prediction compared to just an LSTM model.
Our team is motivated to offer a tool that simplifies investment research and educates users about the dynamics of stock market movements, specifically the movement of the S&P 500 index. Through this, StockGrader.io is poised to play a role in helping novice investors become informed market participants capable of confidently navigating the complexities of the stock market.
In summary, StockGrader.io represents a leap forward in making stock market investing more accessible and understandable for inexperienced investors. This paper will delve further into the technical underpinnings of the LSTM model and Stock Grade system, the design philosophy behind StockGrader.io, and the anticipated benefits of the platform.


2. Literature Review
Many studies have explored the use of machine learning for predicting stock prices. Yet, a common issue they face is the challenge of accurately forecasting the volatile and rapidly changing patterns in stock movements. One study introduces a hybrid approach, combining machine learning and LSTM-based deep learning models for predicting stock prices. This methodology utilizes eight machine learning models, including multivariate linear regression, multivariate adaptive regression spline (MARS), regression tree, bootstrap aggregation (Bagging), extreme gradient boosting (XGBoost), random forest (RF), artificial neural network (ANN), and support vector machine (SVM) alongside four LSTM-based deep learning models. These models focus on multi-step forecasting using univariate and multivariate input data over one or two weeks. Their findings suggest that the most precise model for forecasting the following week's opening price of the NIFTY 50 index is the LSTM-based univariate model that utilizes data from the previous week. 
Conversely, another study critiques the LSTM method, pointing out its weaknesses. Despite LSTMs being adept at identifying patterns over time, which is essential for time-series analysis like stock price prediction, their performance significantly depends on the input data's quality and relevance. The main challenges are the potential overload due to the high dimensionality of financial datasets and the necessity for substantial expertise to determine which features are predictive. Unlike LSTMs, the CNN-LSTM architecture marries the feature extraction capabilities of CNNs with LSTM's sequential data analysis strength, providing a robust framework for stock price prediction. This method excels in processing the complex, noisy, and non-linear data typical of the stock market by efficiently identifying key patterns in historical price data. The combined use of CNNs for initial feature extraction and LSTMs for understanding sequence dependencies enhances prediction accuracy and simplifies the data complexity for the LSTM, potentially reducing training times and computational expenses.
In regards to the development of the composite Stock Grades, much of the inspiration behind its development comes from an industry unrelated to finance: American football. Our team was inspired by ESPN’s “Total QBR” grade. The grade itself is a single value, ranging from 0-100, where the best NFL Quarterbacks would have grades closer to 100 and the worst ones have grades closer to 0 [4]. In ESPN’s Total QBR grade, multiple variables are considered in the algorithm. Utilizing similar logic, our team successfully developed a Stock Grade value ranging from 0 to 1, where a 0 grade indicates a stock should be confidently sold, while a grade closer to 1 indicates a stock should be confidently bought. Like ESPN’s Total QBR grade, multiple variables were weighed and utilized in developing our Stock Grades, intending to reflect a stock’s current market performance. With this Stock Grading system, a novice investor can easily identify a stock that is performing well (and will continue to do so), or a stock that is doing the complete opposite. It is important to note that our Stock Grades are not a total solution to investment decisions but a single, yet powerful tool in the investing toolbox.  
Of course, current stock market information platforms exist, such as Investopedia and CNN Money. However, our tool stands unique in comparison to these existing tools. Investopedia is famed for its ability to explain stock market jargon and mathematical formulas, while CNN Money is noted for its ability to deliver key basic stock market information almost immediately. StockGrader.io remains unique in that basic stock market information is presented daily without sacrificing the educational aspect of learning about stock market terms and algorithms. In other words, our tool serves as an all-in-one location for investors to learn more about the S&P 500 index, and the stock market in general. This is due to the fact that our analysis (with Stock Grades and price prediction with LSTM), along with our detailed descriptive work, serves as a hub for any investor to not only learn about the fundamentals of the stock market, but also detailed analytic information about the S&P 500 index that can be leveraged to make investment decisions.

 


3.  Methodology
3.1) Planning. 
       During the planning phase of the StockGrader.io project, we focused on identifying the project's main goals, key features, technology options, and potential technical and operational risks to ensure the effectiveness and reliability of the system. This phase is critical to the success of the project. The project's core goal is to provide stock market investors, especially beginners and younger-generation investors, with an intuitive and easy-to-understand stock grading and price prediction system, with detailed explanations of relevant jargon. The system aims to simplify complex stock market data in a quantitative way and help investors easily grasp the potential value and risk of stocks. To achieve these goals, we identified key features of the system, including real-time acquisition of stock data, analysis of key financial and market momentum indicators, development of a comprehensive scoring algorithm for stock stability, and design of a user-friendly interactive interface.
      Regarding technology stack selection, we chose the Python language because of its extensive library support and powerful data processing capabilities. In the project, the yfinance library is used to crawl stock data from Yahoo Finance, pandas is used for data manipulation and preprocessing, numpy is used for numerical calculations, and TensorFlow and Keras are used to build and train LSTM models to predict stock price trends.
      The planning phase also includes an assessment of potential risks, such as the accuracy and completeness of the data, the predictive performance of the model, and the complexity of the technology implementation. In addition, we consider compliance and data security requirements to ensure that the system can operate within a legal and compliant framework.
      Finally, we established the project timeline and set key milestones and evaluation points. Resource allocation considers staffing, financial investment, and time management to ensure that projects are completed on time and meet expected quality standards.

3.2) Front-End. 
	With the usage of the Python programming language, the Streamlit framework was utilized to produce a high-quality frontend web application. The usage of Streamlit allowed for the backend portion of our application to easily be fused with the frontend portion, where any user input can be quickly connected to the backend. This makes updating any frontend changes relatively easy, which is necessary given our analytics’ daily updates of stock information (from Yahoo! Finance’s free API). 
	The usage of Streamlit allowed multiple pages to be easily created on the web application without having to produce numerous HTML pages (and corresponding CSS code). This allowed the team additional time to be utilized in backend production, especially for the developments of the price prediction model and the Stock Grading system. 
More specifically, the web application contains three different pages. One page is a “Home” page, where the user can see popular Stock Grades, along with an interactive feature to create their own grades. This allows a user to understand the educational nature of our platform without overwhelming them with a variety of charts, graphs, and additional visualizations. StockGrader.io’s second page is an “Individual S&P 500 Stock Metrics” page. This page is where users can find both basic and advanced information about each stock in the S&P 500 index. This page is where our LSTM charts are provided, as well as an individual Stock Grade highlighting the current performance of that stock. Additionally, a Candlestick chart is provided in conjunction with Momentum and Trend charts, where users can input a time range to investigate. Finally, our third page is the “Definitions and Explanations” page, which serves as the hub for all of the descriptions of every term, chart, and algorithm presented that may not be common knowledge to novice investors. These pages work together in harmony, where users can find descriptive analytics of every S&P 500 stock, while being able to quickly read about any confusing terms, making StockGrader.io serve as an all-in-one hub to learn about the S&P 500 index.
3.3) Back-End (LSTM). 
Integrating current data is crucial to achieving effective stock price prediction [3]. Data collection is performed through a free API from Yahoo Finance, which gathers a dataset encompassing stock details such as open, high, low, close, adjusted close, volume, dividends, and stock splits. The yfinance library retrieves historical stock price data for specific companies over designated periods. Furthermore, we have created a function that verifies the validity of the company symbol (Ticker) by cross-referencing it with a list of S&P 500 companies obtained from Wikipedia.
For the stock price prediction model, we used the LSTM model. LSTM networks are particularly suited for stock price prediction because they can remember long-term dependencies and effectively manage time series data [1]. This capability is vital in the financial domain, where past data significantly influences future trends. Despite these strengths, the complexity and computational demands of LSTMs and challenges in model interpretability remain significant considerations. Thus, while LSTMs offer robust tools for predicting stock prices, they should be integrated into a comprehensive investment strategy that balances technical and fundamental analysis. We, therefore, have integrated adequate data preprocessing, feature engineering, and especially hyperparameter tuning before training the model. 
Considering the fast-changing economic environment, we have set the period of the entire dataset as 3 months. Meanwhile, we have tested different values for the parameters that significantly affect model performance: days used for prediction, number of units in LSTM layers, dropout rate, number of dense units, number of epochs, and batch size. Then the parameters are modified accordingly.
To evaluate the model's performance, we have used Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Correlation Coefficient (r) values. As can be seen from Figure 1, LSTM does not look perfect, but it shows fair performance, with low RMSE and MAE (shown in Table 1).


FIGURE 1: Graph of Actual and Predicted Prices by LSTM


Table 1: MSE, RMSE, MAE, and Correlation Coefficient for LSTM



LSTM
MSE
822.8722
RMSE
28.6857
MAE
23.1519
Correlation Coefficient
-0.3835



3.4) Back-End (Stock Grading). 
      In the back-end development of the stock rating system, the required stock data is first automatically captured from Yahoo Finance through the yfinance library. The data includes a stock's price-to-earnings ratio (P/E Ratio) and relative strength index (RSI), which are key financial and market momentum indicators for evaluating a stock's performance. After the data is obtained, the Pandas library is used to sort and preprocess data to ensure its accuracy and availability.
      The system then uses numpy to perform numerical calculations, including normalizing the P/E ratio and RSI index. The normalization process involves converting the values of these indicators into a range of 0 to 1, making comparisons between different stocks more consistent and fair. This step is achieved by setting the minimum and maximum values for each indicator to ensure the uniformity of the evaluation criteria and the objectivity of the rating.
      The algorithm then combines these normalized indicators to calculate a composite score for each stock. This score is obtained by weighting different indicators, where the distribution of weights is optimized and adjusted based on historical data and market feedback to reflect the actual influence of different indicators on stock stability.
The algorithm sorts the stocks once the scores are calculated based on predefined rating criteria. For example, a stock's composite score within a specific range would be classified as a "Potential Strong Buy (Figure 2)," while one with a lower score might be rated a "Potential Strong Sell." These classifications reflect a stock's overall financial stability and market performance.


FIGURE 2: Example of Stock Grading

      Ultimately, the stock rating system provides a user-friendly interface that displays each stock's rating and score, allowing investors to identify and evaluate potential investment opportunities quickly. This system not only improves the efficiency of investment decisions but also enhances the information support for decision-making. It is especially suitable for beginners and investors who want to simplify the investment process. In this way, the system helps investors better understand and utilize market data and make smarter investment choices.

3.5) Risks. 
The innovative project StockGrader.io, designed to simplify stock market investing for novice investors, particularly those from Generation Z, carries several inherent risks that could impact its success. Firstly, the platform's dependence on the LSTM (and possibly CNN-LSTM) model for stock price predictions raises concerns about the accuracy and reliability of these predictions, as these models might only partially account for sudden market changes or unexpected events. There is also a risk that the simplified grading system could lead users to overlook other crucial financial data and market conditions, leading to over-reliance on the grades provided.
Additionally, handling sensitive financial information necessitates robust data security measures to prevent breaches and protect user privacy, which is crucial for maintaining user trust. The technological complexity and costs associated with implementing and scaling advanced models like LSTM and CNN-LSTM pose further challenges, particularly as the user base grows. Moreover, widespread adoption of the platform could influence market dynamics, potentially diminishing the effectiveness of its predictive models as market participants adjust their behavior based on the grades provided.
Legal and regulatory challenges also loom, as financial advisory services are subject to strict regulations, which could impact how StockGrader.io's recommendations are perceived and used. There is also the potential for the educational components of the platform to be insufficient, which might not fully equip users to understand the complexities of the stock market, potentially leading to poor investment decisions. Finally, rapid technological advancements could render the chosen models obsolete, requiring continuous updates and adaptations to keep the platform current and effective. Addressing these risks is crucial for ensuring the long-term success and reliability of StockGrader.io, enabling it to fulfill its goal of transforming novice investors into informed participants in the stock market

3.6) Logical Testing. 
Case I: Inadequate COMPANY name (ticker) and prediction_days Compared to Available Data. 
An inaccurate ticker name, such as ‘TSLAa’ instead of ‘TSLA’ (Figure 3), would result in an error or generate completely different company information (Figure 4). Modifying the prediction_days variable to a value larger than the available dataset (e.g., Figure 5: setting it to 10000 when the dataset contains fewer days) results in an error (Figure 6), because the code tries to create training data samples that are not possible with the available data points. This could raise an IndexError due to attempts to access indexes beyond the dataset's length.

FIGURE 3: Inaccurate Company Name.


FIGURE 4: Error Message.


FIGURE 5: Large prediction_days.


FIGURE 6: Error Message.


Case II: Very High Epochs and batch size for training.
Increasing the epochs and batch_size parameters in the model.fit() method to extremely high values leads to excessive memory usage or very long training times. For instance, setting epochs to 10,000 (Figure 7) without adequate computational resources could freeze or severely slow down the system, making it unresponsive (Figure 8).

FIGURE 7: model.fit() Function.


FIGURE 8: Training the Model with a Large Number of Epochs.


To address these issues, we have developed a function that verifies the company name by cross-referencing it with the list of S&P 500 stocks obtained from Wikipedia. If the ticker information doesn't match, the function terminates the process and generates an error message automatically managed by our front-end platform. This ensures that we do not proceed with any further analysis if the ticker information is incorrect. For the parameters such as prediction_days, epochs, and batch_size, we've established default values through hyperparameter tuning to optimize performance.

Case III: Invalid date inputs.
Case Three involves a front-end situation where the program crashes if the user provides inputs for date ranges that are not allowed, such as providing a Starting and/or Ending Date that is a future date, as shown in Figure 9. Additionally, the program can crash if the date range has an Ending Date that comes before the Starting Date, as shown in Figure 10. An error message is delivered in these instances to avoid crashing the program, allowing the user to adjust the Date Ranges again before rerunning the program. This is shown in Figure 11, where the error message provides a warning symbol that can capture the user’s attention. But, shown in Figure 12, the error message is not displayed, and the correct information shows up, if the Date Ranges are accurate.

FIGURE 9


FIGURE 10


FIGURE 11


FIGURE 12


3.7) Back-end Unit Testing.
We have implemented several testing procedures to validate the functionality and reliability of our system, focusing on the validity of company tickers, the adequacy of prediction days, and the appropriateness of epochs during training.
The objective of testing validity of company ticker is to ensure that the ‘is_valid_ticker’ function can accurately distinguish between valid and invalid company tickers. For the test cases, we use a known valid ticker, such as ‘TSLA,’ and a known invalid ticker, such as ‘TSLAa.’ The expected result is that the function should return True for a valid ticker and False for an invalid one, effectively preventing errors in further data processing.
To adjust values for parameters, ‘prediction_days’, ‘epochs’, and ‘batch_size’, we have built a separate code file with ‘for loop’ that tests different values for these parameters and other parameters that affect the model performance. Then the parameters are modified accordingly. 

3.8) Front-end Unit Testing.
Since our program operates locally, we haven't encountered accessibility issues, allowing multiple users to access the application simultaneously. Our frontend is powered by Streamlit, which facilitates the transfer of user inputs to our backend host, subsequently refreshing the dashboard based on these inputs. However, it's important to note that while the dashboard is accessible, it cannot update in real-time without an internet connection.


4.  Conclusions
4.1) Pros.
StockGrader.io is designed as an innovative fintech solution aimed at making stock market investing more approachable for novice and Generation Z investors. Leveraging advanced predictive models like the LSTM, and exploring enhancements with CNN-LSTM combinations, the platform provides a numerical grading system for stocks within the S&P 500. This approach not only simplifies investment decisions but also serves to educate users, enhancing their financial literacy and making the stock market less daunting. By focusing on these younger, tech-savvy investors, StockGrader.io taps into a demographic that is keen yet often overwhelmed by traditional investment channels.

4.2) Cons.
However, there are notable challenges and potential drawbacks associated with StockGrader.io. The inherent unpredictability of financial markets can limit the effectiveness of even the most advanced predictive models, such as LSTMs. Users might overly rely on the platform's grades, potentially overlooking other vital investment considerations like market conditions and company fundamentals. Moreover, issues related to data security, privacy, and potential biases in machine learning models could affect the platform’s reliability and user trust. Additionally, widespread use of such predictive tools might influence market behavior itself, possibly diminishing the effectiveness of the models over time due to adaptive market responses.
 
4.3) Challenges.
One major challenge we faced was mastering the intricacies of LSTM models and optimizing their parameters. The effectiveness of these models hinges on several variables, such as the length of the dataset used. Longer datasets may improve prediction accuracy, but the dynamic nature of economic conditions poses a challenge to consistent forecasting. It’s crucial to acknowledge that stock prices are influenced by a myriad array of factors beyond historical data, including company performance, economic trends, technological innovations, shifts in consumer behavior, marketing strategies, and corporate developments. Regrettably, the data we had access to was inadequate in capturing these complexities fully. As a result, predictions based solely on past price trends could be swiftly negated by sudden, unpredictable external events, making the selection of an appropriate timeframe for predictions notably difficult.
Another significant challenge was integrating our team’s LSTM models and stock grading system into Streamlit for our frontend application. Initially, displaying LSTM charts and stock grade tables posed several technical issues, stemming from problems with installing and integrating new Python packages like Tensorflow into our project files. Furthermore, our team members were tasked with distinct responsibilities—some focused on frontend development, others on developing the stock grading system, and some on crafting the LSTM models. The integration of all these different segments into a cohesive application was complicated by variations in the Python versions and the packages installed on individual team members’ machines.
Additionally, time constraints significantly impacted our project development. Although we managed to implement all the core features of our application, we were left wishing for more time to explore additional functionalities, such as implementing CNN-LSTM models. From the start of the semester until now, several aspects of the project demanded more time than we initially expected. This included researching ways to generate stock grades and acquiring a deep understanding of Flask, which later shifted to Streamlit. This extended learning curve highlighted the need for more development time to fully realize our application’s potential

4.4) Lessons Learned. 
This project underscored the formidable challenge of predicting stock prices within a constrained timeframe for several reasons. Stock price volatility is affected by a broad range of external factors that add complexity to forecasting efforts. Beyond just historical price data, various elements such as company performance, economic trends, technological advancements, consumer behavior, marketing success, and corporate occurrences all significantly influence stock prices. Unfortunately, our limited data and tight deadlines prevented a comprehensive incorporation of these factors. Consequently, even predictions based on historical trends can be rendered obsolete by unexpected events, leading to a greater emphasis on short-term or algorithmic trading in recent technological approaches to the stock market. Furthermore, a significant dilemma in stock price prediction is deciding the amount of historical data to utilize. While years of data may be available for well-established companies, managing and analyzing such extensive data across different periods is overwhelming. Identifying the most relevant time frame for reliable predictions is challenging, as more data does not necessarily improve forecast accuracy. This is complicated by changing statistical characteristics over time, which can diminish the relevance of older data for future predictions. The inadequacy of AI models trained on specific periods, such as during the COVID-19 market boom, to adapt to new market conditions illustrates the need for more sophisticated machine learning strategies capable of addressing these challenges.
Another key insight from this project is the realization that understanding new technologies or packages often takes longer than anticipated. In future projects, it is crucial to allocate ample time for researching new technologies, as there is a tendency to underestimate the learning curve involved. This insight extends to understanding industry basics, like the stock market. While numerous resources are available to enhance market knowledge, it is important to invest sufficient time to discern which information is pertinent and which is not. This lesson is universally applicable across various sectors beyond finance, emphasizing the need for thorough preparation and research in any field.



5.  Future Works
For the LSTM and CNN-LSTM models, future enhancements will focus on fine-tuning parameters, especially those concerning time frames. Next, we aim to produce more sophisticated and informative graphs for our frontend platform, StockGrader.io. Our primary objective with StockGrader.io has been to offer fresh perspectives on S&P 500 stocks through straightforward metrics and comprehensive explanations, making it accessible for everyone, regardless of their knowledge of the stock market. We believe we have successfully met this goal. With additional time and resources, there is substantial potential to enhance StockGrader.io, such as incorporating various machine-learning models like CNN-LSTM and refining the accuracy of stock grades by integrating a wider range of metrics, including market volatility. These backend improvements will be seamlessly integrated into our frontend system with user-friendly explanations of the CNN-LSTM models for users from diverse technical backgrounds.
The Stock Grades also have room for improvement, as additional variables and improved weights could be added to improve the ability for the grades to reflect a stock’s performance. Additionally, more Stock Grades can be produced. Specifically, a Grade could be produced that highlights a stock’s stability, which could be valuable for users who look to invest in stocks with high stability. 
The visual appeal of the StockGrader.io application could also be upgraded. Currently, the background is quite basic and could be made more engaging with additional imagery. Moreover, for a selected stock, integrating live news feeds could provide users with real-time updates on significant events affecting that stock.
Currently, StockGrader.io only analyzes stocks from the S&P 500 index. There is potential to expand our scope to include other indices, such as the Dow Jones Industrial Average and the Russell 2000 Index, and to broaden our focus to include international stock indices, which would attract a more global audience. Further development could also include expanding the range of indices we cover, potentially broadening our user base to those interested in various American and international stock indices.
Lastly, there is an opportunity to develop a mobile version of StockGrader.io. Creating iOS and Android apps would extend the functionality of the web-based platform to mobile users. Although this was considered a lower priority due to time constraints, it represents a significant avenue for future expansion.


References
[1]	AjNavneet. (2023.). GitHub - AjNavneet/StockPricePrediction_YFinance_LSTM_RNN: Enhanced stock prices forecasting using Yfinance, LSTM and RNN. GitHub. https://github.com/AjNavneet/StockPricePrediction_YFinance_LSTM_RNN
[2] Katz, Sharon, and Brian Burke. “How Is Total QBR Calculated? We Explain Our Quarterback Rating.” ESPN.com, 8 Sept. 2016, www.espn.com/blog/statsinfo/post/_/id/123701/how-is-total-qbr-calculated-we-explain-our-quarterback-rating.
[3] 	Mehtab, S., Sen, J., Dutta, A. (2021). Stock Price Prediction Using Machine Learning and LSTM-Based Deep Learning Models. In: Thampi, S.M., Piramuthu, S., Li, KC., Berretti, S., Wozniak, M., Singh, D. (eds) Machine Learning and Metaheuristics Algorithms, and Applications. SoMMA 2020. Communications in Computer and Information Science, vol 1366. Springer, Singapore. https://doi.org/10.1007/978-981-16-0419-5_8
[4] 	Lu, W., Li, J., Li, Y., Sun, A., & Wang, J. (2020). A CNN-LSTM-Based Model to Forecast Stock Prices. Complexity, 2020, Article ID 6622927. https://doi.org/10.1155/2020/6622927
 

