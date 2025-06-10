import streamlit as st
import math

class Calculator:
    def __init__(self):
        operators = {
            '+': self.add,
            '-': self.subtract,
            '*': self.multiply,
            '/': self.divide,
        }
        self.operators = operators

    def add(self, num1, num2):
        return num1 + num2

    def subtract(self, num1, num2):
        return num1 - num2

    def multiply(self, num1, num2):
        return num1 * num2

    def divide(self, num1, num2):
        if num2 == 0:
            raise ValueError('Cannot divide by zero')
        return num1 / num2

    def add_operation(self, operator, function):
        return self.operators.update({operator: function})

    def calculate(self, num1, operator, num2=None):
        if operator not in self.operators:
            raise ValueError('Invalid operator, use +, -, *, /, ^, sqrt, log')
        if operator == 'sqrt':
            return self.operators[operator](num1)
        if num1 == '' or num2 == '':
            raise ValueError('Invalid input, should not be empty')
        if not isinstance(num1, (int, float)) or not isinstance(num2, (int, float)):
            raise ValueError('Input should be a number or float')
        return self.operators[operator](num1, num2)

# Initialize the calculator
main_calculator = Calculator()
main_calculator.add_operation('^', lambda x, y: x ** y)
main_calculator.add_operation('sqrt', lambda x: x ** 0.5)
main_calculator.add_operation('log', lambda x, y: math.log(x, y))

# Streamlit app
st.title("Simple Calculator wit python an StreamLit")
st.write("Select an operator and enter numbers to perform calculations.")

# Operator selection
operator = st.selectbox(
    "Select operator:",
    options=['+', '-', '*', '/', '^', 'sqrt', 'log'],
    index=0
)

# Input fields
col1, col2 = st.columns(2)
with col1:
    num1_input = st.text_input("Enter first number:", value="0.0")
with col2:
    if operator != 'sqrt':
        num2_input = st.text_input("Enter second number:", value="0.0")
    else:
        num2_input = None
        st.write("Second number not required for sqrt")

# Calculate button
if st.button("Calculate"):
    try:
        num1 = float(num1_input)
        if operator == 'sqrt':
            result = main_calculator.calculate(num1, operator)
            st.success(f"sqrt({num1}) = {result}")
        else:
            num2 = float(num2_input)
            result = main_calculator.calculate(num1, operator, num2)
            st.success(f"{num1} {operator} {num2} = {result}")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Clear button to reset inputs
if st.button("Clear"):
    st.session_state.num1_input = "0.0"
    if operator != 'sqrt':
        st.session_state.num2_input = "0.0"
    st.rerun()