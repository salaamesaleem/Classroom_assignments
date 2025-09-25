import streamlit as st

st.title('Temperature Converter')

temperature = st.number_input('Enter temperature:')
from_unit = st.radio('Convert from:', ['Celsius', 'Fahrenheit', 'Kelvin'], horizontal=True)
to_unit = st.radio('Convert to:', ['Celsius', 'Fahrenheit', 'Kelvin'], horizontal=True)

if st.button('Convert'):
    if from_unit == to_unit:
        result = temperature
    else:
        if from_unit == 'Celsius':
            if to_unit == 'Fahrenheit':
                result = (temperature * 9 / 5) + 32
            else:  # Kelvin
                result = temperature + 273.15
        elif from_unit == 'Fahrenheit':
            if to_unit == 'Celsius':
                result = (temperature - 32) * 5 / 9
            else:  # Kelvin
                result = (temperature - 32) * 5 / 9 + 273.15
        else:  # Kelvin
            if to_unit == 'Celsius':
                result = temperature - 273.15
            else:  # Fahrenheit
                result = (temperature - 273.15) * 9 / 5 + 32

    st.success(f'{temperature}° {from_unit} = {result:.2f}° {to_unit}')