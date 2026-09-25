# Detecção de Objetos com YOLO: Boi Garantido x Boi Caprichoso

Desafio de código da [DIO](https://www.dio.me) sobre rotulagem de base de dados e treinamento da rede YOLO. Em vez do LabelMe e do Darknet original, usei uma stack mais atual, mas seguindo exatamente a mesma ideia pedida: pegar um detector pré-treinado no COCO e fazer transfer learning para reconhecer classes novas, que o modelo original nunca viu.

## O desafio

Rotular uma base de dados e treinar uma rede YOLO para detecção de objetos, com pelo menos duas classes novas além das já treinadas previamente. O enunciado aceita LabelMe mais Darknet, as imagens já rotuladas do COCO, ou transfer learning no Colab para quem não conseguir rodar a YOLO localmente. Optei por uma quarta via: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), o sucessor direto e mais usado hoje em dia da mesma família de redes, com pesos pré-treinados no COCO prontos para transfer learning local.

## As duas classes novas

`boi_garantido` e `boi_caprichoso`, as mascotes dos dois bois-bumbás do Festival de Parintins. Nenhuma das duas existe nas 80 classes do COCO. As imagens vieram do dataset já construído e curado no projeto [NeuralNetworks-TransferLearning](https://github.com/CROMA-LEARNING/NeuralNetworks-TransferLearning) deste mesmo portfólio, que originalmente foi feito para classificação, não detecção.

## Do dataset de classificação ao dataset de detecção

Aqui apareceu o primeiro problema real do projeto. O dataset de classificação inclui qualquer imagem relacionada ao time, torcida, bandeira, personagens individuais, sem exigir que o boi físico apareça na foto. Para detecção isso não serve: sem o objeto boi na imagem não existe caixa delimitadora para desenhar. Foi necessário filtrar, revisando visualmente as 225 imagens, quais mostram de fato a figura do boi, cabeça, estátua ou costume. Sobraram 56, 23 de Garantido e 33 de Caprichoso.

Com as imagens certas em mãos, faltava a caixa delimitadora em si, que o dataset original não tem. Em vez de rotular manualmente no LabelMe, usei o [rembg](https://github.com/danielgatis/rembg), que roda o modelo pré-treinado U2Net para segmentar o objeto principal da foto, e derivei a caixa a partir da máscara resultante. Uma forma de auto anotação fraca, bem mais rápida que rotular na mão.

O problema é que essa abordagem tem um viés conhecido: modelos de saliência genéricos, como o U2Net, são treinados majoritariamente com fotos de pessoas e tendem a marcar rostos e corpos humanos como o objeto mais saliente da cena, mesmo quando a intenção da foto é outra. Boa parte do dataset tem justamente isso, alguém posando ao lado da mascote, ou dançando com ela, e nesses casos o rembg com frequência marcou a pessoa, não o boi. Revisei manualmente as 56 caixas geradas e descartei todas em que a caixa não isolava razoavelmente a figura do boi. Sobraram 20 imagens, 7 de Garantido e 13 de Caprichoso, divididas em 15 para treino e 5 para validação, com as duas classes representadas nos dois conjuntos.

É um dataset pequeno, e isso aparece nos resultados. Documento os dois lados na seção de resultados.

## Treinamento

Fine-tuning do `yolov8n.pt`, a versão mais leve da família YOLOv8 pré-treinada no COCO, sobre as 20 imagens anotadas. 100 épocas configuradas, com early stopping de paciência 20, imagem em 640px, batch 8. O treino parou sozinho na época 63.

```bash
python scripts/build_dataset.py   # gera dataset/images e dataset/labels a partir das listas aprovadas
python scripts/train.py           # fine-tuning do YOLOv8n
python scripts/predict.py foto.jpg  # roda o detector treinado numa imagem nova
```

## Resultados

![Curvas de treino](results/training_curves.png)

O melhor checkpoint, salvo automaticamente pelo Ultralytics como `best.pt`, ocorreu na época 46, com mAP50 de 0.954 e mAP50-95 de 0.300 no conjunto de validação. São só 5 imagens de validação, então esse número isolado tem que ser lido com cautela, mas a curva de mAP mostra uma evolução real ao longo do treino, não ruído aleatório.

![Matriz de confusão](results/confusion_matrix.png)

Rodando o `best.pt` nas 5 imagens de validação com limiar de confiança 0.4, mantendo só a detecção mais confiante por imagem: as duas imagens de Garantido foram detectadas e classificadas corretamente, com confiança alta, 0.91 e 0.90. Das três imagens de Caprichoso, uma foi classificada corretamente, e duas foram confundidas com Garantido.

![Exemplos de detecção](results/deteccoes_exemplo.png)

O padrão faz sentido. Garantido tem só 7 imagens de treino, todas com o boi branco bem definido contra fundos variados, e o modelo aprendeu essa classe de forma consistente. Caprichoso tem mais exemplos, 13, mas com mais variação de enquadramento, incluindo fotos onde pessoas aparecem perto do boi preto, exatamente o tipo de cena onde a anotação automática já era menos confiável. Com 20 imagens no total, qualquer característica espúria de algumas fotos específicas pesa demais na decisão do modelo.

## O que fica de aprendizado

O ponto mais importante deste projeto não foi o resultado do detector em si, foi descobrir e documentar o motivo real da limitação: anotação automática via segmentação de saliência genérica funciona bem para fotos de objeto único e centralizado, e falha de forma previsível quando há pessoas na cena competindo pela atenção do modelo de segmentação. Isso é uma limitação conhecida de pipelines de anotação fraca na literatura de detecção de objetos, e ver isso acontecer no próprio dataset, com uma causa identificável e não só um número ruim sem explicação, vale mais que treinar em cima de rótulos errados sem perceber.

Para melhorar o detector de verdade, o próximo passo não é mexer no treino, é aumentar e corrigir o dataset: rotular manualmente as imagens que o rembg errou, em vez de descartá-las, e se possível reunir mais fotos de Caprichoso com o boi isolado e bem enquadrado.

## Créditos

- Modelo base: [YOLOv8n](https://github.com/ultralytics/ultralytics), Ultralytics, pré-treinado no COCO.
- Segmentação automática: [rembg](https://github.com/danielgatis/rembg), modelo U2Net.
- Imagens: reaproveitadas e filtradas do dataset do projeto [NeuralNetworks-TransferLearning](https://github.com/CROMA-LEARNING/NeuralNetworks-TransferLearning), fontes e licenças documentadas lá.
