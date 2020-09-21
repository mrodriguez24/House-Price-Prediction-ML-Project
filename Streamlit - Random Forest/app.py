import pandas as pd 
# import plotly.express as px
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mpl_toolkits
from datetime import datetime
import os, time, sys
from pandas import DataFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import tree


st.title('Waka Waka Seattle Home Prices')
st.write("""
Our app predicts the **Seattle House Price**!
""")
st.write('---')

from PIL import Image
img = Image.open("seattle.jpg")
st.image(img)
st.write('---')
st.subheader("""
Home Prices Predicted Using Machine Learning
""")


df = pd.read_csv('clean1.csv')


#read and display csv
@st.cache 
def fetch_data(): 
    df = pd.read_csv('clean1.csv')

    return df


y = df['price']
X = df[['bedrooms',
 'bathrooms',
 'sqft_living',
 'sqft_lot',
 'floors',
 'waterfront',
 'view',
 'condition',
 'yr_built',
 'yr_renovated',
 'zipcode',
 'year',
 'month'
]]

#Select and split X and y
def preprocessing(df):
    X = df.iloc[:,1:].values
    y = df.iloc[:,0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the Random Forest model
@st.cache(allow_output_mutation=True)
def randomForest(X_train, X_test, y_train, y_test):
    regressor = RandomForestRegressor(n_estimators = 100, random_state=42)
    regressor.fit(X_train,y_train)
    y_predict = regressor.predict(X_test)
    score = r2_score(y_test, y_predict)*100

    return score ,regressor

@st.cache(allow_output_mutation=True)
def linearRegression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)*100

    return score , model


#User input for the Random Forest model
def user_input_RF():
    bedrooms = st.slider("Bedrooms: ", 1,15)
    bathrooms = st.text_input("Bathrooms: ")
    sqft_living = st.text_input("Square Feet: ")
    sqft_lot = st.text_input("Lot Size: ")
    floors = st.text_input("Number Of Floors: ")
    waterfront = st.text_input("Waterfront? For Yes type '1',  For No type '0': ")
    view = st.slider("View (A higher score will mean a better view) : ", 0,4)
    condition = st.slider("House Condition (A higher score will mean a better condition): ", 1,5)
    yr_built = st.text_input("Year Built: ")
    yr_reno = st.text_input("A Renovated Property? For Yes type '1',  For No type '0': ")
    zipcode = st.text_input("Zipcode (5 digit): ")
    year_sold = st.text_input("Year Sold: ")
    month_sold = st.slider("Month Sold: ", 1,12)
   
    user_input_prediction = np.array([bedrooms,bathrooms,sqft_living,
    sqft_lot,floors,waterfront,view,condition,yr_built,yr_reno,zipcode,year_sold,month_sold]).reshape(1,-1)
    
    return(user_input_prediction)


#User input for linear regression model
def user_input_LR():
    bedrooms = st.slider("Bedrooms: ", 1,15)
    bathrooms = st.number_input("Bathrooms: ",0.0,10.0,1.0,0.25)
    sqft_living = st.number_input("Square Feet: ",0.0,30000.0,0.0,1.0)
    sqft_lot = st.number_input("Lot Size: ",0.0,30000.0,0.0,1.0)
    floors = st.number_input("Number Of Floors: ",0.0,5.0,1.0,0.25)
    waterfront = st.number_input("Waterfront? For Yes type '1',  For No type '0': ",0.0,1.0,0.0,1.0)
    view = st.slider("View (A higher score will mean a better view) : ", 0,4)
    condition = st.slider("House Condition (A higher score will mean a better condition): ", 1,5)
    yr_built = st.slider("Year Built: ",1900,2020)
    yr_reno = st.slider("A Renovated Property? For Yes type '1',  For No type '0': ",0,1)
    zipcode = st.number_input("Zipcode (5 digit): ",0.0,100000.0,0.0,1.0)
    year_sold = st.number_input("Year Sold: ",0.0,100000.0,0.0,1.0)
    month_sold = st.slider("Month Sold: ", 1,12)
   
    user_input_prediction = np.array([bedrooms,bathrooms,sqft_living,
    sqft_lot,floors,waterfront,view,condition,yr_built,yr_reno,zipcode,year_sold,month_sold]).reshape(1,-1)

    return(user_input_prediction)

def main():
    data = fetch_data()
    X_train, X_test, y_train, y_test = preprocessing(data)

    #Show data
    if st.checkbox("Show the Data We Used"):
        st.subheader("Home Sales From 2014 to 2015")
        st.dataframe(data)
        if st.checkbox("Quick View of Features Histogram"):
            data.hist(bins=50, figsize=(15,15))
            st.pyplot()
    st.write('---')

    #Side bar 
    if(st.sidebar.button("Back to Homepage")):
        import webbrowser 
        webbrowser.open('https://hungnw.github.io/seattle-house-prediction.github.io/')
    st.sidebar.header("Menu")
    st.sidebar.selectbox("Choose a City", ["Seattle"])
    ml_model = st.sidebar.selectbox("Choose a Model to Predict Home Prices", ["Random Forest Regressor","Mutilple Linear Regression","Coming Soon"])
    viz = st.sidebar.selectbox("Visualization",['None','Feature Importance -RF Only','Tree- RF Only'])

    #Add line space for Home buttom 
    # if(st.sidebar.button("Home")):
    #     import webbrowser 
    #     webbrowser.open('https://hungnw.github.io/seattle-house-prediction.github.io/')

    #RF Model 
    if(ml_model == "Random Forest Regressor"):
        st.subheader("Random Forest Model")
        score, regressor= randomForest(X_train, X_test, y_train, y_test)
        txt = "Accuracy of Random Forest Model is: " + str(round(score,2)) + "%" 
        st.success(txt)
        st.write("---")
        
        if(viz == 'Feature Importance -RF Only'):
            feature_names = X
            score, regressor= randomForest(X_train, X_test, y_train, y_test)
            importance = sorted(zip((regressor.feature_importances_), feature_names), reverse=True)
            importance_df = DataFrame(importance, columns=['Feature_importances','Feature_names'])
            importance_df.set_index('Feature_names',inplace=True)
            importance_df.sort_values(by = 'Feature_importances', ascending=True, inplace=True)
            # st.markdown("Randon Forest feature importance")
            # fig = px.bar(importance_df, x='Feature_importances')
            # st.plotly_chart(fig)
            # st.bar_chart(importance_df)
            importance_df.plot(kind='barh')
            plt.title('Random Forest feature importance')
            plt.legend(loc='lower right')
            st.pyplot()
            st.write('---')

        elif(viz == 'Tree- RF Only'): 
            # from sklearn.tree import export_graphviz
            # # Export as dot file
            # export_graphviz(regressor.estimators_[3], 
            #     max_depth=3,
            #     out_file='tree.dot', 
            #     feature_names = list(X.columns),
            #     class_names = data.price,
            #     rounded = True, proportion = False, 
            #     precision = 2, filled = True)
            st.markdown("Qucik view of decision tree no.3")
            tree = open('tree.txt')
            st.graphviz_chart(tree.read(),use_container_width=True)
            st.write('---')
        

        try:
            if(st.checkbox("Start a Search")):
                user_input_prediction = user_input_RF()
                pred = regressor.predict(user_input_prediction)
                error = 94784
                if(st.button("Submit")):
                    st.write('Mean Absolute Error: ', int(error))
                    txt = 'The Predicted Home Price is: $' + str(int(pred)) + ' \u00B1 $' + str(error)
                    st.success(txt)

                feedback = st.radio("Waka Waka value your feedback, please rate from 1-5, (5) being excellent:", ('Please choose from below','1','2','3','4','5'))
                if feedback == 'Please choose from below':
                    st.text('')
                elif feedback == '1': 
                    st.error("This option is not valid")
                elif feedback == '2': 
                    st.warning("We have tried our best!")
                elif feedback == '3': 
                    st.success("Thank you for your feedback.")
                elif feedback == '4': 
                    st.success("Glad you like it!")
                elif feedback == '5':
                    st.success("Waka Waka agrees with you. Have a nice day!")
        except:
            pass
    
    #LR Model 
    if(ml_model == "Mutilple Linear Regression"):
        st.subheader('Linear Regression Model')
        score, model= linearRegression(X_train, X_test, y_train, y_test)
        txt = "Accuracy of Linear Regression Model is: " + str(round(score,2)) + "%. Proceed with caution" 
        st.warning(txt)
        st.write('---')
        

        try:
            if(st.checkbox("Start a Search")):
                user_input_prediction = user_input_LR()
                pred = model.predict(user_input_prediction)
                if(st.button("Submit")):
                    txt = 'The Predicted Home Price is: $' + str(int(pred))
                    st.success(txt)

                feedback = st.radio("Waka Waka value your feedback, please rate from 1-5, (5) being excellent:", ('Please choose from below','1','2','3','4','5'))
                if feedback == 'Please choose from below':
                    st.text('')
                elif feedback == '1': 
                    st.error("This option is not valid")
                elif feedback == '2': 
                    st.warning("We have tried our best!")
                elif feedback == '3': 
                    st.success("Thank you for your feedback.")
                elif feedback == '4': 
                    st.success("Glad you like it!")
                elif feedback == '5':
                    st.success("Waka Waka agrees with you. Have a nice day!")
        except:
            st.write('error')


    #Coming Soon 
    elif(ml_model == "Coming Soon"):
        text = "Coming  Soon..."
        i=0
        while i < len(text):
            st.write(text[i])
            time.sleep(0.3)
            i += 1


if __name__ == "__main__":
	main()