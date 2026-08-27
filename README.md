# Contador de Pessoas - Auditório (protótipo PWA)

Protótipo funcional de um PWA (Progressive Web App) que usa IA de visão computacional
para detectar e contar pessoas em tempo real com **Ultralytics YOLO11n**, usando a câmera do dispositivo — tudo
rodando **dentro do navegador**, sem enviar vídeo para nenhum servidor.

## Como funciona

- A câmera captura o vídeo ao vivo (`getUserMedia`).
- O modelo **Ultralytics YOLO11n**, exportado para ONNX, roda no navegador com ONNX
  Runtime Web e filtra somente a classe `person`.
- A câmera e a inferência só começam ao pressionar **Iniciar**. **Parar** encerra a
  inferência e libera imediatamente a câmera do aparelho.
- Cada pessoa detectada é desenhada como um retângulo sobre o vídeo, e o total é mostrado
  em destaque, junto com o percentual de ocupação (se você definir a capacidade do
  auditório) e um mini-histórico das últimas leituras.
- Um Service Worker (`sw.js`) faz cache dos arquivos e das bibliotecas de IA, permitindo
  instalar o app na tela inicial e reabrir mesmo com conexão instável.
- Você pode ajustar a **sensibilidade** de detecção direto na tela e baixar um
  **relatório em CSV** com o histórico de
  contagens e os eventos de ocupação (70%, 90%, 100%).

## Como rodar localmente

Navegadores só liberam acesso à câmera em contexto seguro (HTTPS) **ou** em `localhost`.
Por isso, não dá para abrir o `index.html` direto como arquivo (`file://`) — é preciso
servir a pasta por HTTP.

Com Python (já vem instalado na maioria dos sistemas):

```bash
cd auditorio-pwa
python3 -m http.server 8080
```

Depois abra **http://localhost:8080** no navegador (Chrome ou Edge recomendados no
desktop; no Android, Chrome; no iPhone, Safari).

Alternativa com Node.js:

```bash
npx serve auditorio-pwa
```

Na primeira vez, o navegador vai pedir permissão de câmera — aceite. O carregamento do
modelo de IA pode levar alguns segundos (ele é baixado pelo PWA na primeira execução e
depois fica em cache).

## Instalando como app

Com a página aberta, o navegador deve oferecer "Instalar app" (ou aparece o botão
"Instalar app" no topo da página). Uma vez instalado, ele abre em tela cheia, sem a
barra de endereço, como um aplicativo nativo.

## Publicando de verdade (produção)

Para usar em um auditório real, valeria publicar em algum serviço com HTTPS gratuito,
por exemplo GitHub Pages, Netlify ou Vercel — basta subir a pasta `auditorio-pwa/`
como está. Aí qualquer celular ou tablet consegue abrir o link e instalar o app.

## Por que a contagem falha ou "pisca" entre valores (ex.: 2 e 0)?

Mesmo com YOLO, em cenas de escritório/auditório reais, várias coisas derrubam a
confiança da detecção: pessoas sentadas, parcialmente
atrás de monitores/mesas/cadeiras, vistas de lado ou de trás, pouca luz, ou câmera muito
próxima da cena. Quando a confiança de uma pessoa fica na borda do limiar, ela some e
aparece de novo a cada leitura — é isso que causa o "piscar" entre 2 e 0 nas suas fotos.

Este protótipo já foi ajustado para atacar isso:

- **YOLO11n a 640×640** substitui o COCO-SSD e preserva mais detalhes de pessoas pequenas,
  sentadas ou parcialmente ocultas, mantendo o processamento viável em celulares.
- **Sensibilidade ajustável**: o controle deslizante define o limiar de confiança
  mínimo (padrão 0.25). Baixe para 0.15–0.20 se
  ainda estiver perdendo gente; suba se estiver contando coisas erradas como pessoa.
- **Suavização temporal**: o número exibido agora é a média das últimas 3 leituras
  (~1s), então uma falha pontual em um único frame não derruba o contador para 0.

Mesmo assim, para um cenário de auditório real, o ganho de precisão mais importante
costuma vir do **posicionamento da câmera**: colocá-la mais alta e angulada para baixo,
enxergando o corpo/tronco das pessoas (não só a cabeça atrás de um monitor), melhora
muito a taxa de acerto. Em salas muito cheias e com bastante sobreposição, o teto de
precisão de qualquer detector de objetos (não só este) é limitado — o próximo passo
seria um modelo de *crowd counting* por estimativa de densidade (ex.: CSRNet) rodando
em um backend, treinado especificamente para multidões dessas.

## Relatório de ocupação

A cada leitura, o app guarda um registro (horário, contagem, % de ocupação) na memória
do navegador. Sempre que a ocupação cruza 70%, 90% ou 100% da capacidade definida, um
evento aparece na lista "Relatório" na tela. O botão **Baixar CSV** exporta todo esse
histórico da sessão atual em um arquivo `.csv` (colunas: `data_hora`, `pessoas`,
`ocupacao_pct`), pronto para abrir no Excel/Sheets. O botão **Limpar** zera o registro
para começar uma nova sessão. Como é tudo local (sem servidor), feche a aba só depois de
baixar o CSV, senão o histórico se perde.

## Atualizando uma instalação já publicada (GitHub Pages, Netlify, etc.)

Como o app usa Service Worker para cache, se você já tinha instalado/aberto a versão
antiga, o navegador pode continuar servindo os arquivos velhos. Depois de subir esta
nova versão, force uma atualização: feche todas as abas do app, reabra e dê um
"recarregar forçado" (no Chrome Android: menu ⋮ → configurações do site → limpar
dados do site; no desktop: Ctrl/Cmd+Shift+R). O `sw.js` desta versão já foi marcado
com uma nova versão de cache (`v4-yolo`) para ajudar nessa troca.

## Outras limitações deste protótipo

- **Cobertura do ambiente**: uma única câmera de celular dificilmente cobre um
  auditório inteiro. Em uso real, vale uma câmera fixa grande-angular no fundo da sala,
  ou múltiplas câmeras com contagem combinada.
- **Desempenho em celulares antigos**: se travar, aumente o `DETECTION_INTERVAL_MS` em
  `app.js` para detectar com menos frequência.
- **Privacidade**: o protótipo não faz reconhecimento facial nem identifica pessoas —
  apenas conta. Nenhum frame de vídeo sai do dispositivo.

## Estrutura de arquivos

```
auditorio-pwa/
├── index.html      # página principal
├── style.css        # estilos
├── app.js           # câmera + IA + lógica de contagem
├── sw.js             # service worker (cache/offline)
├── manifest.json     # metadados do PWA
├── icons/             # ícones do app
└── README.md
```
