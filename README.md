# Previsão de Prêmios de Seguros - Kaggle Competition 🏆

Este repositório contém o script desenvolvido para uma competição de dados no Kaggle com o objetivo de prever os valores dos prêmios de seguros com base nas variáveis preditoras fornecidas.

🚀 Funcionalidades
	1.	Análise Estatística
Foram aplicadas técnicas de análise exploratória e estatística para compreender a relação entre as variáveis preditoras e a variável-alvo.
	2.	Pré-processamento de Dados
	•	Normalização da variável-alvo usando a transformação Box-Cox para estabilizar a variância e tornar os dados mais próximos de uma distribuição normal.
	•	Aplicação de transformações logarítmicas conforme as exigências do desafio.
	3.	Modelagem
Foram utilizados três algoritmos diferentes, cada um com ajustes de parâmetros específicos:
	•	Decision Tree Regressor
	•	XGBoost Regressor
	•	LightGBM Regressor
	4.	Métrica de Avaliação
A métrica escolhida foi o Root Mean Squared Error (RMSE), que mede a diferença entre os valores previstos e os reais. O RMSE penaliza grandes erros e é amplamente utilizado para problemas de regressão.
Fórmula do RMSE: 
￼
Onde:
	•	yᵢ = Valor real (observado).
	•	ŷᵢ = Valor previsto (pelo modelo).
	•	n = Número total de observações.
	•	Σ = Somatório (soma de todos os erros ao quadrado).
	•	√ = Raiz quadrada.

🛠️ Tecnologias Utilizadas

As bibliotecas e dependências estão listadas no arquivo requirements.txt contido neste repositório. Certifique-se de instalar todos os pacotes antes de executar o script.

📂 Estrutura do Repositório
	•	script.py: Código principal contendo a análise, pré-processamento e modelagem.
	•	requirements.txt: Lista de bibliotecas e suas versões necessárias para a execução do projeto.
	•	LICENSE: Licença de uso do repositório.

⚙️ Como Usar
	1.	Clone este repositório:

git clone https://github.com/sturaro-ds/ml_insurance_predictions.git


	2.	Instale as dependências:

pip install -r requirements.txt


	3.	Execute o script:

python script.py

📊 Resultados

Os modelos foram avaliados com base na métrica RMSE, e os resultados obtidos para cada algoritmo estão documentados no código.

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

Se precisar ajustar algo ou incluir mais detalhes, é só avisar! 🚀
