import streamlit as st
import pandas as pd

from pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(
    page_title= "Student Churn Predictor",
    page_icon = "🎓",
    layout= "wide"
)

st.title("🎓 Student Churn Predictor")
st.markdown("Fill in student details to predict **dropout risk** using Machine Learning")
st.divider()

st.subheader("👤 Personal Information")
col1, col2, col3 = st.columns(3)

with col1:
    marital_status = st.selectbox(
        "Maritial Status",
        options=[1,2,3,4,5,6],
        format_func=lambda x:{
            1:"Single", 2:"Married", 3:"Widower",
            4:"Divorced", 5:"Common law", 6:"Legally Separated"
        }[x]
    )
    gender = st.selectbox(
        "Gender",
        options=[1,0],
        format_func=lambda x: "Male" if x==1 else "Female"
    )

with col2:
    age_at_enrollment = st.number_input(
        "Age at Entrollement",min_value=17, max_value=70, value=20
    )
    nationality = st.number_input("Nationality Code", value=1)

with col3:
    international = st.selectbox(
        "International Student",
        options=[0,1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    displaced = st.selectbox(
        "Displaced",
        options=[0,1],
        format_func=lambda x: "No" if x==0 else "Yes"
    )

st.divider()

st.subheader("💰 Financial Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    debtor = st.selectbox(
        "Debtor",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:
    tuition_fees_up_to_date = st.selectbox(
        "Tuition Fees Up To Date",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:
    scholarship_holder = st.selectbox(
        "Scholarship Holder",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col4:
    educational_special_needs = st.selectbox(
        "Special Educational Needs",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

st.divider()

st.subheader("📋 Application Details")
col1, col2, col3 = st.columns(3)

with col1:
    application_mode = st.number_input("Application Mode", value=1)
    application_order = st.number_input(
        "Application Order", min_value=0, max_value=9, value=1
    )

with col2:
    course = st.number_input("Course Code", value=9254)
    attendance = st.selectbox(
        "Attendance",
        options=[1, 0],
        format_func=lambda x: "Daytime" if x == 1 else "Evening"
    )

with col3:
    admission_grade = st.number_input(
        "Admission Grade", min_value=0.0, max_value=200.0, value=142.5
    )

st.divider()

st.subheader("🏫 Previous Qualification & Family")
col1, col2, col3 = st.columns(3)

with col1:
    previous_qualification = st.number_input(
        "Previous Qualification", value=1
    )
    previous_qualification_grade = st.number_input(
        "Previous Qualification Grade",
        min_value=0.0, max_value=200.0, value=160.0
    )

with col2:
    mothers_qualification = st.number_input(
        "Mother's Qualification", value=1
    )
    fathers_qualification = st.number_input(
        "Father's Qualification", value=1
    )

with col3:
    mothers_occupation = st.number_input(
        "Mother's Occupation", value=5
    )
    fathers_occupation = st.number_input(
        "Father's Occupation", value=5
    )

st.divider()

st.subheader("📚 1st Semester Performance")
col1, col2, col3 = st.columns(3)

with col1:
    cu1_credited = st.number_input("Units Credited (S1)", value=0)
    cu1_enrolled = st.number_input("Units Enrolled (S1)", value=6)

with col2:
    cu1_evaluations = st.number_input("Evaluations (S1)", value=8)
    cu1_approved = st.number_input("Units Approved (S1)", value=5)

with col3:
    cu1_grade = st.number_input(
        "Grade (S1)", min_value=0.0, max_value=20.0, value=13.5
    )
    cu1_without_evaluations = st.number_input(
        "Without Evaluations (S1)", value=0
    )

st.divider()

st.subheader("📚 2nd Semester Performance")
col1, col2, col3 = st.columns(3)

with col1:
    cu2_credited = st.number_input("Units Credited (S2)", value=0)
    cu2_enrolled = st.number_input("Units Enrolled (S2)", value=6)

with col2:
    cu2_evaluations = st.number_input("Evaluations (S2)", value=8)
    cu2_approved = st.number_input("Units Approved (S2)", value=5)

with col3:
    cu2_grade = st.number_input(
        "Grade (S2)", min_value=0.0, max_value=20.0, value=13.5
    )
    cu2_without_evaluations = st.number_input(
        "Without Evaluations (S2)", value=0
    )

st.divider()

st.subheader("📊 Economic Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    unemployment_rate = st.number_input(
        "Unemployment Rate (%)", value=10.8
    )
with col2:
    inflation_rate = st.number_input(
        "Inflation Rate (%)", value=1.4
    )
with col3:
    gdp = st.number_input("GDP", value=1.74)

st.divider()

if st.button("🔍 Predict Dropout Risk", use_container_width=True):
    try:
        data = CustomData(
            marital_status=marital_status,
            application_mode=application_mode,
            application_order=application_order,
            course=course,
            attendance=attendance,
            previous_qualification=previous_qualification,
            previous_qualification_grade=previous_qualification_grade,
            nationality=nationality,
            mothers_qualification=mothers_qualification,
            fathers_qualification=fathers_qualification,
            mothers_occupation=mothers_occupation,
            fathers_occupation=fathers_occupation,
            admission_grade=admission_grade,
            displaced=displaced,
            educational_special_needs=educational_special_needs,
            debtor=debtor,
            tuition_fees_up_to_date=tuition_fees_up_to_date,
            gender=gender,
            scholarship_holder=scholarship_holder,
            age_at_enrollment=age_at_enrollment,
            international=international,
            cu1_credited=cu1_credited,
            cu1_enrolled=cu1_enrolled,
            cu1_evaluations=cu1_evaluations,
            cu1_approved=cu1_approved,
            cu1_grade=cu1_grade,
            cu1_without_evaluations=cu1_without_evaluations,
            cu2_credited=cu2_credited,
            cu2_enrolled=cu2_enrolled,
            cu2_evaluations=cu2_evaluations,
            cu2_approved=cu2_approved,
            cu2_grade=cu2_grade,
            cu2_without_evaluations=cu2_without_evaluations,
            unemployment_rate=unemployment_rate,
            inflation_rate=inflation_rate,
            gdp=gdp
        )

        pred_df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        prediction, probability = pipeline.predict(pred_df)

        st.divider()

        if prediction[0] == 1:
            st.error(
                f"⚠️ Student is Likely to Drop Out  "
                f"| Dropout Probability: "
                f"{probability[0][1]*100:.1f}%"
            )
        else:
            st.success(
                f"✅ Student is Likely to Continue  "
                f"| Continuation Probability: "
                f"{probability[0][0]*100:.1f}%"
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Dropout Probability",
                f"{probability[0][1]*100:.1f}%"
            )
        with col2:
            st.metric(
                "Continuation Probability",
                f"{probability[0][0]*100:.1f}%"
            )

    except Exception as e:
        st.error(f"Error: {e}")
