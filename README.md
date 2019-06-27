

# Detecção de Orelhas em Imagens 2D Utilizando Redes Neurais Convolucionais  

Este repositório possui três diretórios, além de quatro outros arquivos. O arquivo *tcc.pdf* contém o texto que detalha o processo de construção do detector, utilizado como trabalho de conclusão do curso de bacharelado em ciência da computação da Universidade Federal da Bahia. Os outros arquivos são utilizados para a extração e preparação da base de dados usada no trabalho. O diretório *detector* possui a implementação da rede, junto com os códigos para treino, teste e cálculo dos resultados. O diretório *pretrained\_detector* possui o mesmo conteúdo, porém conta com os resultados calculados. O modelo pré-treinado, utilizado no trabalho, também está disponível e será explicado como é possível obtê-lo. Detalhes acerca dos arquivos contidos nesses diretórios e como manuseá-los serão mostrados nas instruções a seguir.
  
## Requerimentos

+ Python3 
+ Tensorflow v1.6.0
+ OpenCV-Python v3.4.3.18
+ NumPy v1.15.4
+ Matplotlib v3.0.2
+ Imgaug v0.2.6  

## Instruções de uso  
  

### Como obter a base de dados utilizada e as marcações do trabalho  
  

1. Executar o *script download\_and\_extract\_data.sh* dentro do diretório corrente. Este *script* fará o *download* e extração da base de dados utilizada no trabalho.  
2. Executar o *script*  *preprocess\_and\_merge\_markings.sh* para redimensionar as imagens para 160x160 e substituir as marcações originais pelas marcações contidas no diretório *tcc\_markings*.   
  

Ao final da execução dos *scripts*, será criado um diretório *database*, contendo todas as imagens de treino, validação e teste, com as marcações manuais complementares feitas durante o nosso trabalho, e no tamanho a ser utilizado pela rede.   

### Obter os modelos pré-treinados utilizados no trabalho (caso queira treinar o detector e gerar um novo modelo, esse passo não é necessário)

1. Baixar o arquivo *models_and_logs_ear_detection.zip* no link <p align="center">[https://drive.google.com/open?id=1fDq8bAcwY4RoiOR-H89I5My1fJGqtfa7](https://drive.google.com/open?id=1fDq8bAcwY4RoiOR-H89I5My1fJGqtfa7)  </p> O arquivo contém os modelos e *logs* do treinamento das duas redes utilizadas, que foram utilizados para o cálculo dos resultados finais. 
2. Extrair os arquivos dentro do diretório *pretrained_detector*  

### Como treinar o detector (caso queira utilizar o modelo pré-treinado, essa parte pode ser pulada) 
  

1. Entrar no diretório *detector* após ter obtido a base de dados e executado o pré-processamento.  
2. Executar o *script train.<span></span>py* para treinar o preditor da rede, implementado no arquivo *network.<span></span>py*.   
	 + O subdiretório *logs* contém informações do *loss* e acurácia do modelo do preditor que são calculados durante o treino a cada época utilizando a base de validação. Para visualizar, é necessário executar o *tensorboard* utilizando o seguinte comando: <p align="center">tensorboard --logdir='./logs' --port=7373 </p> O *tensorboard* será acessível a partir do endereço: <p align="center">[http://localhost:7373/](http://localhost:7373/)  </p> O uso da porta 7373 é apenas um exemplo e não é necessário o uso desse mesmo valor.  O modelo do preditor será salvo no subdiretório *model*. 
3. Executar o *script class\_train.py* para treinar o classificador da rede, implementado no arquivo *class_network.<span></span>py*.  
	 + O subdiretório *class\_logs* contém as mesmas informações anteriores para o classificador. Para visualizar, é necessário executar o *tensorboard* utilizando o seguinte comando: <p align="center"> tensorboard --logdir='./class\_logs' --port=3737 </p> O *tensorboard* será acessível a partir do endereço: <p align="center">[http://localhost:3737/](http://localhost:3737/)  </p>  Da mesma forma aqui, é possível utilizar outro número para a porta. O modelo do classificador será salvo no subdiretório *class_model*.

O arquivo *data_manager.<span></span>py* contém a implementação de funções utilizadas para gerenciar as imagens e marcações durante o treino, teste e inferência. As redes são independentes e podem ser treinadas em paralelo. Após esses passos, os modelos do classificador e do preditor terão sido gerados e poderão ser usados para inferência. 

### Cálculo das métricas na base de dados (caso não esteja retreinando a rede, este passo pode ser pulado)

1. De posse da rede treinada na etapa anterior, execute o *script evaluate.<span></span>py*. Este *script* irá gerar os arquivos *fdr_fnr_results.txt*e *FDRxFNR_Graph.png*. O primeiro contém resultados numéricos da quantidade de verdadeiros positivos (TP), falsos positivos (FP) e falsos negativos (FN), a taxa de falsa descoberta (FDR) e a taxa de falsos negativos (FNR). Estes valores são calculados para diferentes limiares confiança e de IOU, especificados no arquivo. O segundo mostra esses resultados visualmente através de um gráfico com curvas de FDRxFNR  variando o limiar de confiança para cada limiar de IOU.
2. Para calcular os resultados da rede a nível de píxel e gerar imagens marcadas utilizando a base de teste, execute o *script test.<span></span>py* passando como argumento um número entre 0 e 100, indicando o limiar de confiança desejado, como por exemplo: <p align="center"> python3 test.<span></span>py 42 </p> O número 42 indica que queremos utilizar este valor como limiar de confiança para nossas detecções. Esta etapa pode ser realizada antes da anterior, porém a execução do *script* anterior nos dá uma forma de encontrar o melhor limiar de confiança a ser utilizado. Ao final da execução, será criado um arquivo *pixel_wise_results.txt*, que contém as métricas calculadas a nível de *pixel*. Além disso, o subdiretório *test_outputs*, inicialmente vazia, será preenchida com as imagens de teste no formato *outX_IOU.png*, onde *X* é um identificador dado à imagem durante a execução do *script* e *IOU* representa a métrica a nível de *pixel* para aquela imagem com quatro casas decimais. O número da imagem não possui relação com o identificador real na base, e só é utilizado para eliminar a possibilidade de imagens com mesmo *IOU* serem sobrescritas. 
**OBS: caso o subdiretório *test_outputs* seja deletada, ao executar o _script_ as imagens serão processadas, porém não serão salvas. É necessário que o subdiretório esteja presente para que as imagens sejam salvas** 

### Realizando a inferência em outras imagens

Após treinar uma nova rede por completo, ou utilizando a rede pré-treinada, podemos realizar a inferência em imagens de quaisquer tamanho. Para isso, dentro do diretório *detector* (ou *pretrained_detector*) execute o *script inference.<span></span>py*, passando como argumento o diretório onde as imagens se encontram e o limiar de confiança desejado. Como um exemplo, suponha que queremos realizar a inferência em nossa base de teste. Podemos realizar essa tarefa com o seguinte comando <p align="center"> python3 inference.<span></span>py ../database/CollectionB 42 </p> 
As saídas sobrescreverão as imagens originais do diretório, desenhando as regiões detectadas. Imagens de quaisquer tamanho podem ser utilizadas, todas são redimensionadas durante a execução apenas para realizar a inferência, e as detecções são redimensionadas para o tamanho original antes de serem desenhadas. 
**OBS: O _script_ aceita apenas imagens no formato .png, ignorando todos os outros arquivos dentro do diretório passado como argumento (incluindo os subdiretórios) .**



