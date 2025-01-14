## FedSketch: Aprendizado Federado com Privacidade e Eficiência de Comunicação

Este repositório contém a implementação do **FedSketch**, um algoritmo inovador para aprendizado federado que visa **aumentar a privacidade dos dados e a eficiência da comunicação**. O FedSketch utiliza **estruturas de dados probabilísticas chamadas "sketches"** para comprimir os vetores de peso dos modelos de aprendizado de máquina, reduzindo significativamente a quantidade de dados transmitidos entre o servidor e os clientes durante o processo de treinamento.

### Principais Características

*   **Compressão de Modelos:** O FedSketch compacta os modelos de aprendizado de máquina usando sketches, **reduzindo o tamanho dos dados transmitidos em até 73 vezes.**
*   **Privacidade Diferencial:** A natureza probabilística dos sketches garante um alto nível de privacidade diferencial, com valores ϵ atingindo 10^-6, protegendo os dados dos clientes contra ataques de inferência.
*   **Eficiência de Comunicação:** A compressão de modelos resulta em uma comunicação mais eficiente entre o servidor e os clientes, tornando o FedSketch ideal para ambientes com restrições de largura de banda.
*   **Acurácia Preservada:** Apesar da compressão, o FedSketch mantém uma acurácia comparável ao aprendizado federado tradicional, especialmente em conjuntos de dados como MNIST.

### Arquitetura

O FedSketch segue a arquitetura cliente-servidor padrão do aprendizado federado, com as seguintes etapas principais:

1.  **Treinamento Local:** Cada dispositivo da federação treina uma rede neural usando apenas os dados locais presentes naquele dispositivo.
2.  **Compressão e Transmissão:** Os parâmetros da rede neural treinada localmente são compactados em um sketch e transmitidos para o servidor de agregação.
3.  **Agregação:** O servidor de agregação agrega os sketches recebidos dos clientes para criar um modelo global.
4.  **Descompressão e Atualização:** O servidor de agregação compartilha o modelo global com os dispositivos, que descompactam o sketch e retreinam seus modelos locais com novos dados.

### Benefícios

*   **Redução de Custos de Comunicação:** A compressão de modelos reduz significativamente os custos de comunicação, tornando o aprendizado federado mais viável em cenários do mundo real.
*   **Melhoria da Privacidade:** A privacidade diferencial inerente aos sketches garante a proteção dos dados dos clientes, tornando o FedSketch adequado para aplicações sensíveis à privacidade.
*   **Escalabilidade:** O FedSketch pode ser facilmente escalonado para um grande número de clientes, tornando-o adequado para aplicações de aprendizado federado em larga escala.

### Aplicações

O FedSketch pode ser aplicado em vários domínios onde a privacidade dos dados e a eficiência da comunicação são cruciais, incluindo:

*   **Saúde:** Treinamento de modelos de aprendizado de máquina em registros médicos eletrônicos distribuídos sem comprometer a privacidade do paciente.
*   **Finanças:** Detecção de fraudes e outros modelos de aprendizado de máquina em dados financeiros distribuídos.
*   **Internet das Coisas:** Treinamento de modelos de aprendizado de máquina em dispositivos IoT com recursos limitados.

### Limitações

*   O FedSketch pode levar mais tempo para convergir em comparação com o aprendizado federado tradicional em certos conjuntos de dados.
*   A escolha ideal dos parâmetros do sketch (número de hashmaps e tamanho das tabelas de hash) depende do conjunto de dados e requer experimentação.

### Trabalhos Futuros

*   Investigar métodos para otimizar os parâmetros do sketch automaticamente com base nas características do conjunto de dados.
*   Explorar a integração do FedSketch com outras técnicas de preservação de privacidade, como criptografia homomórfica.
*   Avaliar o desempenho do FedSketch em cenários mais complexos, como aprendizado federado com dados não-IID.

### Conclusão

O FedSketch é uma abordagem promissora para o aprendizado federado que aborda as principais preocupações de privacidade e eficiência de comunicação. Ao compactar os modelos usando sketches, o FedSketch reduz significativamente a quantidade de dados transmitidos, preservando a acurácia do modelo e protegendo a privacidade dos dados. As aplicações potenciais do FedSketch são vastas, abrangendo vários domínios onde a privacidade e a eficiência são cruciais.

