import os 
import streamlit as st
from model import final_result , get_pdf_page_as_image

DB_FAISS_PATH = 'vectorstores/'
pdf_path = 'data/Gale_encyclopedia_of_medicine_vol_1.pdf'


# Streamlit webpage title
st.title('Medical Chatbot')

# User input
user_query = st.text_input("Please enter your question:")

# Button to get answer
if st.button('Get Answer'):
    if user_query:
        # Call the function from your chatbot script
        response = final_result(user_query)
        if response:
            # Displaying the response
            st.write("### Answer")
            st.write(response['result'])

            # Displaying source document details if available
            if 'source_documents' in response:
                st.write("### Source Document Information")
                for doc in response['source_documents']:
                    # Retrieve and format page content by replacing '\n' with new line
                    formatted_content = doc.page_content.replace("\\n", "\n")
                    st.write("#### Document Content")
                    st.text_area(label="Page Content", value=formatted_content, height=300)

                    # Retrieve source and page from metadata
                    source = doc.metadata['source']
                    page = doc.metadata['page']
                    st.write(f"Source: {source}")
                    st.write(f"Page Number: {page+1}")
                    
                    # Display the PDF page as an image
                    pdf_page_image = get_pdf_page_as_image(pdf_path, page)
                    st.image(pdf_page_image, caption=f"Page {page+1} from {source}")
                    
        else:
            st.write("Sorry, I couldn't find an answer to your question.")
    else:
        st.write("Please enter a question to get an answer.")
