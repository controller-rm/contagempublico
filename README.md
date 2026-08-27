# Contador de Pessoas - Auditório (protótipo PWA)

Protótipo funcional de um PWA (Progressive Web App) que usa IA de visão computacional
para detectar e contar pessoas em tempo real, usando a câmera do dispositivo — tudo
rodando **dentro do navegador**, sem enviar vídeo para nenhum servidor.

## Como funciona

- A câmera captura o vídeo ao vivo (`getUserMedia`).
- O modelo **coco-ssd** (TensorFlow.js), carregado via CDN, roda no navegador e detecta
  objetos a cada ~300ms, filtrando apenas a classe `person`.
- Cada pessoa detectada é desenhada como um retângulo sobre o vídeo, e o total é mostrado
  em destaque, junto com o percentual de ocupação (se você definir a capacidade do
  auditório) e um mini-histórico das últimas leituras.
- Um Service Worker (`sw.js`) faz cache dos arquivos e das bibliotecas de IA, permitindo
  instalar o app na tela inicial e reabrir mesmo com conexão instável.

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
modelo de IA pode levar alguns segundos (ele é baixado da CDN na primeira execução e
depois fica em cache).

## Instalando como app

Com a página aberta, o navegador deve oferecer "Instalar app" (ou aparece o botão
"Instalar app" no topo da página). Uma vez instalado, ele abre em tela cheia, sem a
barra de endereço, como um aplicativo nativo.

## Publicando de verdade (produção)

Para usar em um auditório real, valeria publicar em algum serviço com HTTPS gratuito,
por exemplo GitHub Pages, Netlify ou Vercel — basta subir a pasta `auditorio-pwa/`
como está. Aí qualquer celular ou tablet consegue abrir o link e instalar o app.

## Limitações deste protótipo (e como evoluir)

- **Multidões densas**: detecção por objeto (coco-ssd) perde precisão quando há muita
  sobreposição de pessoas. Para salas muito cheias, o caminho é um modelo de *crowd
  counting* por estimativa de densidade (ex.: CSRNet) rodando em um backend.
- **Cobertura do ambiente**: uma única câmera de celular dificilmente cobre um
  auditório inteiro. Em uso real, vale uma câmera fixa grande-angular no fundo da sala,
  ou múltiplas câmeras com contagem combinada.
- **Desempenho em celulares antigos**: o modelo `lite_mobilenet_v2` foi escolhido por
  ser leve; em aparelhos muito fracos, aumente o `DETECTION_INTERVAL_MS` em `app.js`
  para detectar com menos frequência.
- **Privacidade**: o protótipo não faz reconhecimento facial nem identifica pessoas —
  apenas conta. Nenhum frame de vídeo sai do dispositivo.
- **Precisão**: ajuste `SCORE_THRESHOLD` em `app.js` (padrão 0.5) para equilibrar
  falsos positivos e falsos negativos conforme a iluminação do ambiente.

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
