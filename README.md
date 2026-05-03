# mesi-face

Implementação em C++ do projeto **FACE (Face Analysis for Commercial Entities)** descrito no artigo:

**Maria De Marsico, Michele Nappi, Daniel Riccio, Harry Wechsler**  
*Robust Face Recognition for Uncontrolled Pose and Illumination Changes*  
IEEE Transactions on Systems, Man, and Cybernetics: Systems, 43(1), 2013.

Este repositório nasceu no contexto da minha **tese de mestrado**, com o objetivo de estudar e implementar um pipeline de reconhecimento facial robusto a variações de **pose** e **iluminação**.

## Contexto

O trabalho original procurava reproduzir o método `FACE`, que combina:

- análise de qualidade da amostra facial
- normalização de pose
- normalização de iluminação
- comparação por correlação local
- índices de fiabilidade da decisão final

O foco do projeto era aproximar a implementação prática do artigo e explorar o comportamento do método em imagens não controladas.

## Estado Atual

O projeto foi modernizado para correr em **Windows** e **Visual Studio 2026**, mantendo a estrutura conceptual do artigo.

Atualmente inclui:

- `face_cli`: executável em linha de comandos
- `face_gui`: interface gráfica Win32 para testes visuais
- build com `CMake`
- dependências geridas por `vcpkg`
- integração com `OpenCV 4`

O pipeline atual executa:

1. deteção facial
2. estimativa de landmarks faciais
3. cálculo de `SP` (pose quality)
4. cálculo de `SI` (illumination quality)
5. normalização de pose
6. normalização de iluminação (`SQI / matching view`)
7. matching por correlação local
8. cálculo de `SRR1` e `SRR2`

## Nota Importante sobre Fidelidade ao Artigo

Este repositório está funcional e executável, mas a implementação ainda não deve ser tratada como uma reprodução científica perfeita do artigo.

As principais limitações atuais são:

- os landmarks ainda não são extraídos com o mesmo mecanismo usado no protótipo original
- a normalização de pose foi reconstruída com base na lógica do código antigo e na descrição do paper, mas ainda precisa de calibração
- a visualização de `SQI` é útil para análise, mas não corresponde a uma “imagem bonita”; é uma representação intermédia para matching

Ou seja: o projeto está num estado **funcional e exploratório**, adequado para estudo, debugging e continuação da tese, mas ainda com margem para refinamento se o objetivo for máxima fidelidade académica.

## Estrutura do Projeto

```text
.
├── data/                    # imagens de exemplo e cascades Haar
├── src/
│   ├── app_support.*        # utilitários partilhados por CLI e GUI
│   ├── face_pipeline.*      # deteção, qualidade, pose, iluminação
│   ├── face_matcher.*       # correlação, ranking, SRR
│   ├── face_gui.cpp         # interface gráfica Win32
│   └── main.cpp             # executável CLI
├── CMakeLists.txt
├── CMakePresets.json
├── launch.vs.json
└── vcpkg.json
```

## Requisitos

- Windows
- Visual Studio 2026
- CMake
- componente C++ instalada no Visual Studio

As dependências de terceiros são resolvidas via `vcpkg` através do ficheiro `vcpkg.json`.

## Compilar no Visual Studio 2026

O projeto está preparado para Visual Studio 2026 através do preset:

- `vs2026-x64`

Configuração manual por linha de comandos:

```powershell
cmake -S . -B build\vs2026-msvc -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE="C:/Program Files/Microsoft Visual Studio/18/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build\vs2026-msvc --config Release
```

Depois disso, os executáveis ficam em:

- `build\vs2026-msvc\Release\face_cli.exe`
- `build\vs2026-msvc\Release\face_gui.exe`

## Usar a CLI

### Identify

```powershell
$env:PATH='C:\Work\mesi-face\build\vs2026-msvc\vcpkg_installed\x64-windows\bin;' + $env:PATH
.\build\vs2026-msvc\Release\face_cli.exe identify --gallery-dir data --query-image .\testface.jpg --cascade-dir data --output-dir output
```

### Train

```powershell
.\build\vs2026-msvc\Release\face_cli.exe train --gallery-dir data --cascade-dir data --output-dir output
```

### Batch

```powershell
.\build\vs2026-msvc\Release\face_cli.exe batch --gallery-dir data --query-dir data --cascade-dir data --output-dir output
```

## Usar a GUI

```powershell
$env:PATH='C:\Work\mesi-face\build\vs2026-msvc\vcpkg_installed\x64-windows\bin;' + $env:PATH
.\build\vs2026-msvc\Release\face_gui.exe
```

A interface permite:

- escolher a galeria
- escolher a imagem de query
- ativar/desativar a normalização de iluminação
- ver a imagem original
- ver a imagem com pose normalizada
- ver a vista `SQI / Matching View`
- consultar métricas e ranking

## Outputs

Os artefactos gerados são gravados na pasta definida em `--output-dir`.

Exemplos:

- `normalized/*_pose.png`
- `normalized/*_normalized.png`
- `normalized/*_sqi.png`
- `histograms/`
- `batch_results.csv`

## Próximos Passos Recomendados

Se o objetivo for aproximar ainda mais a implementação ao artigo original, os próximos passos naturais são:

1. substituir os landmarks heurísticos por um detector facial de landmarks mais fiável
2. recalibrar a normalização de pose face às figuras e fórmulas do paper
3. rever quantitativamente os índices `SP`, `SI`, `SRR1` e `SRR2`
4. validar o sistema com um dataset mais próximo do usado na investigação

## Motivação Académica

Este projeto representa uma parte importante do trabalho desenvolvido na minha tese de mestrado.  
Mais do que uma simples aplicação, é também um registo do processo de investigação, implementação e interpretação prática de um artigo científico na área de biometria e reconhecimento facial.

