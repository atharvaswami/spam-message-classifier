import streamlit as st
import joblib

def predict(text):
    labels = ['Not Spam', 'Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is: '+labels[v])

if __name__ == '__main__':    
    cv = joblib.load("vector.pkl")
    model = joblib.load('model.pkl')

    st.title('Spam Classifier')
    st.image('spam.jpg')
    user_input = st.text_input('Write your message: ')
    submit = st.button('Predict')
    if submit:
        answer = predict([user_input])
        st.text(answer)
