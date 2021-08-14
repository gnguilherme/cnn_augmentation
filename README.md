![MKT](https://img.shields.io/badge/language-Python-orange.svg)

# Criação de CNN simples

Dentro de `src/config.json` está a configuração para o código. Pontos importantes são:

- `model_path`: o modelo irá salvar/carregar os pesos treinados nesse diretório.
Pode apontar para qualquer diretório. Aconselho mudar apenas o `teste` para
apontar para outra pasta, mas ainda dentro da pasta `models`. Obs.: o _tensorflow_
já irá criar automaticamente o diretório `models` (ou qualquer um que for colocado
no `config.json`) durante o treino;
- `input_shape`: tamanho da imagem. Tamanhos comuns: `224x224`, `128x128`, `64x64`. 
Utilize 3 para usar _augmentation_ nas cores (_hue augmentation_); 
- `conv_filters`: quantidade de filtros. Coloque quantos quiser, mas quanto mais filtros,
mais poder computacional será necessário para o treino;

---

Construa o dataset dentro do diretório `data`

```
projeto
   └───data
        ├───train
        ├───validation
        └───test
```

Para executar os códigos: dentro da _root_, execute:

- Para treinar o modelo, execute `python -m src.train`
- Para testar o modelo, execute `python -m src.test_model`. Obs.: Caso queira
executar o modelo utilizando imagens carregadas de outra forma que não pelo
`tf.data.Dataset`, utilize `model.predict(img)`. Não esqueça de:
  - dividir a imagem por `255` antes de entrar com ela na rede;
  - utilizar `np.expand_dims` para transformar a primeira dimensão em _batch_,
como é pedido pelo _tensorflow_. Ex.: `img = np.expand_dims(img, 0)`
  - o modelo vai retornar uma probabilidade de a imagem pertencer à classe alvo.
Valores acima de `.5` quer dizer class `1`