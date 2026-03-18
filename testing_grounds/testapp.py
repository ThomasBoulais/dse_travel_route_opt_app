import streamlit as st

st.header('Scrolling Columns')

cols = st.columns(3)

cols[0].write('A short column')
cols[1].write('Meow' + ' meow'*1000)
cols[2].write('Another short column')

css='''
<style>
    section.main>div {
        padding-bottom: 1rem;
    }
    [data-testid="column"]>div>div>div>div>div {
        overflow: auto;
        height: 70vh;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)