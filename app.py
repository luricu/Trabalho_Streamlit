import pandas as pd
import streamlit as st
import joblib 


# loading the trained model.
model = joblib.load('model/modelo-final-LR.pkl')

# carregando uma amostra dos dados.
dataset = pd.read_csv('data/StudentsPerformance.csv') 
#classifier = pickle.load(pickle_in)

# título
st.title("Data App - Análise Nota Matemática")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição da nota matemática.")

st.sidebar.subheader("Defina os atributos do aluno para predição da nota matemática")

# mapeando dados do usuário para cada atributo
Genero              = st.sidebar.selectbox("Genero"                     , ("Feminino", "Masculino"))
Etnia               = st.sidebar.selectbox("Etnia"                      , ("Grupo A", "Grupo B", "Grupo C", "Grupo D", "Grupo E"))
Escolaridade_Pais   = st.sidebar.selectbox("Escolaridade dos Pais"      , ("Grau de associado","Diploma de bacharel","Ensino médio","Mestrado","Faculdade","Escola secundária"))
Almoco              = st.sidebar.selectbox("Almoço"                     , ("Padrão", "Grátis/Reduzido"))
Curso_Preparacao    = st.sidebar.selectbox("Curso Preparação para Teste", ("Nenhum", "Concluído"))
Pontuacao_Leitura   = st.sidebar.number_input("Nota Leitura"            , value=dataset["Pontuacao_Leitura"].mean())
Pontuacao_Escrita   = st.sidebar.number_input("Nota Redação"            , value=dataset["Pontuacao_Escrita"].mean())

# transformando o dado de entrada em valor binário
if Genero == "Masculino":
  Genero   = 1  
else:
  Genero   = 0  

if Etnia == "Grupo A": 
  Etnia   = 0
if Etnia == "Grupo B":  
  Etnia   = 1
if Etnia == "Grupo C":  
  Etnia   = 2
if Etnia == "Grupo D":  
  Etnia   = 3
else:  
  Etnia   = 4

if Escolaridade_Pais == "Grau de associado": 
  Escolaridade_Pais   = 0
if Escolaridade_Pais == "Diploma de bacharel":  
  Escolaridade_Pais   = 1
if Escolaridade_Pais == "Ensino médio":  
  Escolaridade_Pais   = 2
if Escolaridade_Pais == "Mestrado":  
  Escolaridade_Pais   = 3
if Escolaridade_Pais == "Faculdade":  
  Escolaridade_Pais   = 4
else:  
  Escolaridade_Pais   = 5

if Almoco == "Padrão":
  Almoco   = 0  
else:
  Almoco   = 1  

if Curso_Preparacao == "Nenhum":
  Curso_Preparacao   = 0  
else:
  Curso_Preparacao   = 1  

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()

    data_teste["Genero"]            = [Genero]
    data_teste["Etnia"]             = [Etnia]    
    data_teste["Escolaridade_Pais"] = [Escolaridade_Pais]
    data_teste["Almoco"]            = [Almoco]	
    data_teste["Curso_Preparacao"]  = [Curso_Preparacao]
    data_teste["Pontuacao_Leitura"] = [Pontuacao_Leitura]
    data_teste["Pontuacao_Escrita"] = [Pontuacao_Escrita]

    #imprime os dados de teste    
    print(data_teste)

    #realiza a predição
    result = model.predict(data_teste)
    
    st.subheader("A nota é:")
    result = str(round(result[0],2))
    
    st.write(result)

#no prompt do anaconda, ir no diretorio da api_streamlit : C:\Users\gusta\Meu Drive\Colab Notebooks\DML\api_streamlit>

#rodar o comando: streamlit run app.py