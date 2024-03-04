import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

st.title("PaperCheck")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    uploaded_file1 = col1.file_uploader("Upload AnswerKey", type=["pdf"])
    if uploaded_file1 is not None:
        st.write(f"First PDF file uploaded: {uploaded_file1.name}")

    uploaded_file2 = col1.file_uploader("Upload Answer Sheet of Student", type=["pdf"])
    if uploaded_file2 is not None:
        st.write(f"Second PDF file uploaded: {uploaded_file2.name}")

if st.button("Process PDFs"):

    st.markdown("<hr>", unsafe_allow_html=True)
    with pdfplumber.open(uploaded_file1) as keypdf:
        textkey = ""

        for page in keypdf.pages:
            page_text = page.extract_text()
            if textkey:
                textkey += "\n"
            textkey += page_text


    with pdfplumber.open(uploaded_file2) as keypdf:
        text = ""

        for page in keypdf.pages:
            page_text = page.extract_text()
            text += page_text

    lines = textkey.split("\n")
    questions = []
    answers = []
    current_block = []

    def process_block(current_block):
        block_text = "\n".join(current_block)
        if any(keyword.lower() in block_text.lower() for keyword in ["question", "q:"]):
            questions.append(block_text)
        elif any(
            keyword.lower() in block_text.lower()
            for keyword in ["ans", "answer:", "a:"]
        ):
            answers.append(block_text)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(
            keyword.lower() in line.lower()
            for keyword in ["question", "q:", "ans", "answer:", "a:"]
        ):
            if current_block:
                process_block(current_block)
                current_block = []

        current_block.append(line)

    if current_block:
        process_block(current_block)

    def extract_questions_max_marks_from_strings(strings):
        questions = []
        max_marks = []

        max_marks_pattern = r"\[(\d+)\]"

        for string in strings:
            match = re.search(max_marks_pattern, string)
            if match:
                max_mark = int(match.group(1))
                max_marks.append(max_mark)

                question = re.sub(max_marks_pattern, "", string).strip()
                questions.append(question)
            else:
                max_marks.append(None)
                questions.append(string.strip())

        return questions, max_marks

    questions_clean, max_marks = extract_questions_max_marks_from_strings(questions)

    def replace_newline_with_space(strings):
        replaced_strings = []
        for string in strings:
            replaced_string = string.replace("\n", " ")
            replaced_strings.append(replaced_string)
        return replaced_strings

    ans_final = replace_newline_with_space(answers)
    que_final = replace_newline_with_space(questions_clean)

    def extract_answers(text):
        answers = text.split("Answer")[1:]

        answers = [answer.strip() for answer in answers]
        return answers

    answers_std = extract_answers(text)
    std_ans_final = replace_newline_with_space(answers_std)
    #
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    original_embeddings = model.encode(ans_final, convert_to_tensor=True)
    huggingface_scores = []

    for i, student_answer in enumerate(std_ans_final, start=1):
        student_embedding = model.encode(student_answer, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(
            original_embeddings[i - 1 : i], student_embedding
        ).item()

        huggingface_scores.append(similarity_score * max_marks[i - 1])

    output_file = "answers.txt"

    with open(output_file, "w") as file:
        file.write("Answer Key:\n")
        for i, original_answer in enumerate(ans_final, start=1):
            file.write(f"{que_final[i-1]}\n")
            file.write(f"{original_answer}\n")
        file.write("\n")

        file.write("Answer Sheet:\n")
        for i, student_answer in enumerate(std_ans_final, start=1):
            file.write(f"Answer {student_answer}\n")

    api_key = "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"
    client = OpenAI(api_key=api_key)

    usecase_Topic = """ You are an assistant that is checking and grading the answer sheet of students by comparing it with the answers provided by the teacher which is the answer key. You can also give marks, according to your knowledge but only when you are sure about it! \n You give to the point accurate marks to the students from 0.00 to 1.00 in the format of 'Score' that is a floating point number.
    So get in the form of a numbered list of Questions and Score as specified in the format above."""

    prompt = """ I will provide you the Answer key first marked as 'Answer key' and then the answers of student as 'Student answer:' which will contains a list of 'Question' and its corresponding 'Answer'.
    I want you to compare the answer of student to the corresponding answerkey provided, semantically and  give marks between 0.00 to 1.00 to each answer. Also give marks according to the length of the answer and the total marks. """

    testinput1 = """ Answer Key:\n\nQuestion 1: What is Object-Oriented Programming (OOP)?\nAnswer1: Object-Oriented Programming (OOP) is a programming paradigm that organizes code into objects, which are instances of classes. It emphasizes the concept of "objects" that encapsulate data and behavior. OOP principles include encapsulation, inheritance, and polymorphism, providing a modular and organized approach to software development.\nQuestion 2: Explain Encapsulation in OOP.\nAnswer2: Encapsulation is the OOP principle that involves bundling data (attributes) and methods (functions) that operate on the data into a single unit known as an object. It helps in hiding the internal details of an object and restricting direct access to its implementation. Encapsulation enhances code organization, reduces complexity, and promotes data integrity by controlling access to the internal state of objects.\nQuestion 3: What is Inheritance in OOP?\nAnswer3: Inheritance is a fundamental OOP concept that allows a new class (subclass or derived class) to inherit attributes and behaviors from an existing class (superclass or base class). This promotes code reusability and establishes a hierarchy of classes. The subclass can extend or override the functionality of the superclass while inheriting its characteristics.\nQuestion 4: Describe the concept of Polymorphism in OOP.\nAnswer4: Polymorphism in OOP refers to the ability of objects to take on multiple forms or the ability of a method to perform different actions based on the object it is acting upon. This can be achieved through method overloading (same method name, different parameters) and method overriding (same method name and parameters, different implementation). Polymorphism enhances flexibility and adaptability in code design.\nQuestion 5: What is Abstraction in the context of Object-Oriented Programming?\nAnswer5: Abstraction is the process of simplifying complex systems by modeling classes based on the essential properties and behaviors relevant to the problem at hand, while ignoring unnecessary details. It involves creating abstract classes or interfaces that define a common structure without specifying the implementation. Abstraction helps in managing complexity, improving code readability, and facilitating code maintenance.\nAnswer Sheet:\n\nAnswer1: Object-Oriented Programming (OOP) is a programming paradigm that utilizes the concept of objects to structure code. Objects encapsulate data and behavior, allowing for a more organized and modular approach to software development. OOP principles, such as encapsulation, inheritance, and polymorphism, facilitate code reuse, maintainability, and the modeling of real-world entities.\nAnswer2: Encapsulation is when you put your data and functions in a box (object) so that they stay together. It's like keeping your stuff in a container, so you don't mess things up.\nAnswer3: Inheritance is like when a new class can borrow things from an old class. It's kinda like passing down traits or features from a parent to a child. Makes it easier to reuse code and saves time.\nAnswer4: Polymorphism is when you can do different things with the same method. It's like having one remote control that works for different devices, each button doing a unique thing. It makes code more flexible.\nAnswer5: Abstraction is like hiding the complicated parts and just showing the simple stuff. It's like using a TV remote without knowing all the technical details. Makes it easier to understand and use.\n"""
    testout1 = """1.00\n 0.64\n 0.82\n 0.75\n 0.80\n"""

    testinput2 = """ 
    Answer Key:
    Question 1:
    What is the kind of pain and ache that the poet feels ?
    Answer 1:
    The poet (here poetess) is deeply attached to her mother who is pretty aged, weak and pale. She is troubled
    to think that the old mom might depart in her absence

    Question 2:
    Why are the young trees described as 'sprinting' ?
    Answer 2:
    The young trees running spiritedly stand in sharp contrast to the aged and pale looking mother. The trees
    symbolise youth and life, whereas the old mother is slipping towards the grave.

    Question 3:
    Why has the poet brought in the image of the merry children spilling out of their homes ?
    Answer 3:
    The little children are full of life, hope and cheerfulness. They have just begun life and have a long way to go.
    The old and weak mother of the poetess, however, is fast losing hold on life. She could breath her last any
    day in near future. The image of cheerful children makes the sight of the mother all the more painful.

    Answer Sheet:

    Ans1: poet is worried that her mother may die soon, hence she is in pain.
    Ans2: that is because to symbolise youth and life as he mom is dieing.
    Ans3: poet brought the image as she was feeling pain.
    """
    testout2 = """0.48\n 0.50\n 0.12\n"""

    with open("answers.txt", "r") as file:
        papercheck = file.read()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": usecase_Topic},
            {"role": "user", "content": prompt + testinput1},
            {"role": "assistant", "content": testout1},
            {"role": "user", "content": prompt + testinput2},
            {"role": "assistant", "content": testout2},
            {"role": "user", "content": prompt + papercheck},
        ],
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.2,
        presence_penalty=0,
    )

    result = response.choices[0].message.content
    scores = re.findall(r"\b\d+\.\d+\b", result)

    scores = [float(score) for score in scores]
    for i in range(0, len(scores)):
        scores[i] = scores[i] * float(max_marks[i])

    final_avg_score = []
    for i in range(0, len(scores)):
        final_avg_score.append((scores[i] + huggingface_scores[i]) / 2)

    def convert_to_specific_range(value):
        value = round(value, 2)
        integer_part = int(value)
        fractional_part = value - integer_part

        if 0 <= fractional_part <= 0.25:
            return integer_part
        elif 0.25 < fractional_part < 0.75:
            return integer_part + 0.5
        elif 0.75 <= fractional_part <= 1:
            return integer_part + 1
        else:
            return value

    def convert_array_to_specific_range(input_array):
        result_array = [convert_to_specific_range(value) for value in input_array]
        return result_array

    final_score = convert_array_to_specific_range(final_avg_score)

    report = ""

    for i, (score, max_mark, que, std_ans) in enumerate(
        zip(final_score, max_marks, que_final, std_ans_final), start=1
    ):
        report += f"Score:<bold> {score}/{max_mark} </bold> <br>"
        report += f" {que}<br>"
        report += f"Answer {std_ans}<br><br>"
    
    total_marks = sum(max_marks)
    marks_obtained = sum(final_score)
    percentage = (marks_obtained / total_marks) * 100
    percentage_str = f"{percentage:.2f}%"
    with st.container():
        st.markdown(
            f'<div style="background-color:#000000;padding:16px;border-radius:10px;"><h4>Total Marks: <h1>{marks_obtained} / {total_marks}</h1><h4> Percentage: <h1>{percentage_str}</h1> </h4><p>{report}</p></div>',
            unsafe_allow_html=True,
        )
