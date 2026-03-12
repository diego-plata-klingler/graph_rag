# Analisis Tecnico del Graph RAG

## Objetivo de este documento

Este documento describe el funcionamiento real del sistema RAG del proyecto tal y como esta implementado ahora mismo en el codigo.

La idea es explicar:

1. La arquitectura tecnica completa.
2. Como se parsea la ley y se convierte en datos estructurados.
3. Como se generan los chunks.
4. Como se construye el grafo en Neo4j.
5. Como funciona el retrieval en cada modo.
6. Como se genera la respuesta final.
7. Cuando se buscan vecinos y cuando no.
8. Un ejemplo paso a paso desde la pregunta del usuario hasta la respuesta.

No es una explicacion teorica de RAG en abstracto. Es una explicacion del pipeline real de este proyecto.

---

## 1. Vision general del sistema

El sistema sigue esta idea general:

1. Se parte de un PDF de la Ley 22/2014.
2. Se extrae texto y se reconstruye su estructura juridica:
   - Titulo
   - Capitulo
   - Seccion
   - Articulo
   - Disposicion
3. Cada articulo y disposicion se transforma a un markdown autocontenido.
4. Ese markdown se divide en chunks estructurados.
5. Se crea un grafo en Neo4j con:
   - nodos `Articulo`
   - nodos `Chunk`
   - relaciones jerarquicas
   - referencias cruzadas entre articulos
   - entidades y conceptos
6. En consulta, el sistema recupera articulos/chunks relevantes mediante:
   - articulo directo
   - expansion de query
   - fulltext
   - vector search
   - expansion de grafo
   - vecinos locales
7. El contexto recuperado se pasa al LLM.
8. El sistema sintetiza una respuesta, valida citas y hace un groundedness check.

En otras palabras:

`PDF -> estructura legal -> markdown -> chunks -> grafo Neo4j -> retrieval -> contexto -> LLM -> respuesta`

---

## 2. Piezas tecnicas del proyecto

Las piezas principales son estas:

### 2.1 Extraccion y parsing

- `01_extraer_pdf.py`
- `chunking_utils.py`

Se encargan de:

- extraer texto del PDF
- limpiar ruido
- detectar titulos/capitulos/secciones/articulos/disposiciones
- enriquecer articulos con entidades y conceptos
- generar chunks estructurados
- guardar todo en `data/articulos.json`

### 2.2 Construccion del grafo

- `02_construir_grafo.py`

Se encarga de:

- crear nodos en Neo4j
- crear indices y constraints
- crear nodos `Chunk`
- enlazar `Articulo -> Chunk`
- crear referencias cruzadas entre articulos
- guardar embeddings de chunks

### 2.3 Consulta y respuesta

- `03_consultar.py`

Se encarga de:

- expandir la query
- clasificar el tipo de pregunta
- recuperar contexto
- rerankear resultados
- ampliar vecinos locales
- construir el contexto final
- sintetizar la respuesta con el LLM
- validar citas y groundedness

---

## 3. Fase 1: extraccion y reconstruccion de la ley

### 3.1 Que hace `01_extraer_pdf.py`

El script:

1. extrae texto bruto del PDF
2. limpia saltos, artefactos y ruido
3. detecta la estructura legal
4. acumula texto dentro del articulo o disposicion abierta
5. cierra cada bloque cuando aparece el siguiente
6. deduplica articulos repetidos por numero
7. enriquece entidades/conceptos
8. genera chunks estructurados
9. guarda todo en `data/articulos.json`

### 3.2 Como detecta articulos

La deteccion se hace por regex sobre cada linea del texto legal ya limpiado.

Cuando encuentra algo del tipo:

```text
Artículo 41. Concepto y objeto social
```

crea un objeto de articulo con:

- `id`
- `numero`
- `titulo`
- `titulo_padre`
- `capitulo_padre`
- `seccion_padre`
- buffer temporal de texto

Despues sigue acumulando lineas dentro de ese articulo hasta que aparece otro articulo o una nueva unidad estructural.

### 3.3 Que datos guarda por articulo

Cada articulo queda con al menos:

- identificador normalizado, por ejemplo `art_41`
- numero visible, por ejemplo `41`
- titulo, por ejemplo `Concepto y objeto social`
- texto completo
- contexto jerarquico
- referencias a otros articulos
- entidades mencionadas
- conceptos mencionados

### 3.4 Deduplicacion

Si el parser detecta varias versiones del mismo numero de articulo, conserva la que tenga mas texto.

Esto evita quedarse con una captura truncada de un articulo si el PDF produjo fragmentos duplicados.

---

## 4. Chunking: como se hace realmente

## 4.1 Idea general del chunking

El sistema actual no chunkea por pagina.

Tampoco usa un splitter ciego que trocea cada N tokens sin respetar estructura.

Hace esto:

1. toma cada articulo como unidad juridica principal
2. lo convierte en markdown autocontenido
3. divide ese markdown respetando encabezados y cuerpo legal
4. si sigue siendo demasiado grande, lo subdivide con overlap

Esto significa que el chunking es:

- estructurado
- orientado a articulos
- con contexto jerarquico embebido en el chunk
- con solape controlado

## 4.2 Como se prepara el markdown de un articulo

Antes de chunkear, cada articulo se convierte en algo parecido a esto:

```md
#### Artículo 41. Concepto y objeto social
Título: TÍTULO II
Capítulo: CAPÍTULO I
Sección: Sin sección

1. Las SGEIC son sociedades anónimas o de responsabilidad limitada...
2. Cada ECR y EICC tendrá una sola gestora...
3. La SGEIC será responsable de garantizar...
4. Las SGEIC se regirán por lo previsto en esta Ley...
```

Esto es importante porque el chunk no solo contiene el texto del articulo, sino tambien:

- numero de articulo
- titulo del articulo
- titulo/capitulo/seccion de contexto

Con eso, el embedding y el fulltext trabajan con mas informacion juridica que el simple cuerpo.

## 4.3 Parametros actuales del chunking

Los parametros actuales son:

- `CHUNK_MAX_TOKENS = 350`
- `CHUNK_OVERLAP = 60`

Interpretacion:

- si un bloque entra en 350 tokens aproximados, se queda entero
- si no entra, se divide
- cuando se divide, conserva hasta 60 tokens aproximados del final del chunk anterior

## 4.4 Como decide donde partir

El chunker intenta partir en este orden:

1. por encabezados markdown
2. por parrafos
3. por lineas
4. por frases

Eso reduce el riesgo de romper listas legales o mezclar mitades de apartados sin continuidad.

## 4.5 Como adapta el texto legal antes de trocear

El texto legal se prepara antes del chunking para introducir saltos utiles en:

- apartados numerados
- listas con letras
- ordinales

Eso hace que un articulo con estructura:

```text
1. ... a) ... b) ... c) ...
```

termine mas cerca de:

```text
1. ...
a) ...
b) ...
c) ...
```

y por tanto el splitter tenga fronteras mas limpias.

## 4.6 Ejemplo real de chunking

Tomemos el articulo 41, que en los datos actuales genera 3 chunks.

### Entrada estructurada

```md
#### Artículo 41. Concepto y objeto social
Título: TÍTULO II
Capítulo: CAPÍTULO I
Sección: Sin sección

1. Las SGEIC son sociedades anónimas o de responsabilidad limitada...
2. Cada ECR y EICC tendrá una sola gestora...
3. La SGEIC será responsable de garantizar...
4. Las SGEIC se regirán por lo previsto en esta Ley...
```

### Salida real aproximada

```text
art_41_chunk_000
- incluye cabecera del artículo
- incluye contexto jerárquico
- incluye apartados 1 y 2

art_41_chunk_001
- vuelve a incluir la cabecera del artículo
- incluye el apartado 3
- puede conservar parte del solape si era necesario

art_41_chunk_002
- vuelve a incluir la cabecera del artículo
- incluye el apartado 4
```

Eso explica por que un mismo articulo puede producir 1, 2 o 3 chunks:

- articulo corto -> 1 chunk
- articulo mediano -> 2 chunks
- articulo largo -> varios chunks

### Ejemplos reales actuales en `data/articulos.json`

- Art. 26 -> 2 chunks
- Art. 30 -> 1 chunk
- Art. 41 -> 3 chunks

---

## 5. Fase 2: construccion del grafo

## 5.1 Nodos principales

El grafo usa principalmente estos nodos:

- `Articulo`
- `Chunk`
- `Titulo`
- `Capitulo`
- `Seccion`
- `Entidad`
- `Concepto`
- `Disposicion`

## 5.2 Relaciones principales

Las relaciones mas importantes son:

- `(:Articulo)-[:PERTENECE_A]->(:Titulo|Capitulo|Seccion)`
- `(:Articulo)-[:TIENE_CHUNK]->(:Chunk)`
- `(:Disposicion)-[:TIENE_CHUNK]->(:Chunk)`
- `(:Articulo)-[:REFERENCIA]->(:Articulo)`
- `(:Articulo)-[:MENCIONA]->(:Entidad)`
- `(:Articulo)-[:TRATA_SOBRE]->(:Concepto)`

## 5.3 Indices

Se crean tres indices clave:

### Fulltext sobre articulo

Permite buscar por:

- titulo del articulo
- texto del articulo

### Fulltext sobre chunk

Permite buscar por:

- texto del chunk
- titulo del articulo padre
- titulo/capitulo/seccion padre

### Vector index sobre chunk

Permite retrieval semantico a nivel chunk.

Esto es importante: el vector search no se hace sobre el articulo entero, sino sobre chunks.

## 5.4 Embeddings

Los embeddings se guardan en cada `Chunk`.

Eso significa:

- mas granularidad en la recuperacion
- mejor recall semantico para conceptos concretos
- posibilidad de volver a agregar despues a nivel articulo

---

## 6. Fase 3: retrieval

Esta es la parte mas compleja del sistema actual.

El retrieval no es una sola busqueda. Es un pipeline de varias capas.

## 6.1 Modos de consulta

Hay tres modos:

### Modo `grafo`

Hace retrieval manual:

- expansion de query
- semillas juridicas
- fulltext
- expansion por grafo
- reranking
- vecinos locales

### Modo `vector`

Hace retrieval semantico:

- embedding de la query
- vector search sobre chunks
- fallback textual si hace falta
- expansion por grafo
- reranking
- vecinos locales

### Modo `cypher`

Hace retrieval via LLM -> Cypher:

- expande la query
- el LLM genera Cypher
- Neo4j devuelve filas
- si hace falta, añade retrieval directo/fallback textual

No todos los modos recuperan exactamente igual, pero comparten piezas comunes.

---

## 7. Expansion de query

Antes del retrieval, la query puede expandirse con:

### 7.1 Sinonimos legales

Ejemplo:

- `SGEIC` -> `sociedad gestora`, `sociedades gestoras`
- `ECR` -> `entidades de capital-riesgo`, `capital riesgo`

### 7.2 Hints de titulos de articulos

Ejemplo:

- `que es` -> `concepto`
- `definicion` -> `definición`
- `actividad` -> `actividad principal`

La expansion solo se usa para retrieval.

La pregunta original se sigue usando en la fase final de generacion.

Ejemplo:

```text
Pregunta original:
¿Qué significa SGEIC?

Query expandida para retrieval:
¿Qué significa SGEIC? sociedad gestora sociedades gestoras concepto
```

---

## 8. Clasificacion de la pregunta

Antes de recuperar, el sistema clasifica la pregunta como:

- `articulo_directo`
- `definicion`
- `comparativa`
- `condiciones`
- `general`

Esto afecta a:

- que articulos se siembran
- como se priorizan titulos
- que articulos se meten completos en contexto

Ejemplos:

- `¿Qué dice el artículo 26?` -> `articulo_directo`
- `¿Qué significa SGEIC?` -> `definicion`
- `¿Qué diferencia hay entre SCR y FCR?` -> `comparativa`
- `¿Qué requisitos tiene una SGEIC?` -> `condiciones`

---

## 9. Retrieval en modo grafo

El flujo real del modo `grafo` es este:

1. Detectar articulos directos en la query.
2. Expandir la query.
3. Generar semillas juridicas.
4. Buscar entidades mencionadas.
5. Ejecutar fulltext sobre chunks.
6. Ejecutar fulltext sobre articulos.
7. Si la pregunta es definicional/comparativa, reforzar con articulos de titulos como:
   - Definicion
   - Regimen juridico
   - Concepto
   - Objeto
8. Expandir por subgrafo:
   - referencias
   - misma seccion
   - mismo capitulo
9. Rerankear a nivel articulo.
10. Recuperar vecinos locales.
11. Recortar al maximo de articulos de contexto.

### 9.1 Que significa "unidad primaria = chunk"

Cuando el fulltext o el vector search devuelven un chunk, el sistema:

1. localiza el articulo padre
2. adjunta ese chunk como `chunk_relevante`
3. agrega varios chunks sobre el mismo articulo
4. rerankea ya a nivel articulo

Asi evita contestar con un chunk aislado sin saber a que articulo pertenece.

### 9.2 Reranking

El reranking mezcla:

- score previo del retrieval
- solapamiento lexical con la pregunta
- señales estructurales del titulo del articulo
- prioridad de articulos directos
- prioridad de semillas

Si esta disponible un reranker real, se usa encima de eso.

Si no, se queda con el heuristico.

---

## 10. Retrieval en modo vector

El modo vector hace esto:

1. expande la query
2. calcula embedding de la query expandida
3. busca en `chunk_vector`
4. agrega resultados a nivel articulo
5. si los resultados son flojos o escasos, complementa con `buscar_articulos`
6. añade articulos directos/semilla
7. expande por subgrafo
8. rerankea
9. añade vecinos locales
10. pasa a generacion

Eso significa que el modo vector no es un vector search "puro".

Es un vector search hibrido, porque si hace falta entra tambien el retrieval textual.

---

## 11. Vecinos locales: cuando se buscan y cuando no

Esta es una capa extra reciente.

## 11.1 Que son los vecinos locales

Hay dos tipos:

### Vecinos de articulo

Si un articulo top es, por ejemplo, el 41, se consideran articulos vecinos:

- 39
- 40
- 42
- 43

En general:

`articulo_base ± 2`

### Vecinos de chunk

Si dentro del articulo 41 el chunk mas relevante es el `orden = 1`, se consideran vecinos:

- chunk orden 0
- chunk orden 2
- y si existieran, tambien los de distancia 2

En general:

`chunk_orden_base ± 2`

## 11.2 Cuando si se buscan

Se buscan en estos casos:

- cuando `buscar_articulos` ya ha recuperado resultados y hay articulos top claros
- cuando `_recuperar_articulos_objetivo` resuelve un articulo directo
- cuando `consulta_vector` ya ha recuperado articulos/chunks relevantes

## 11.3 Cuando no se buscan

No se buscan:

- sobre todos los articulos recuperados, solo sobre los top
- si no hay articulos suficientes sobre los que anclar la expansion local
- sobre filas crudas de Cypher como texto libre, salvo que ademas entre el contexto directo o el fallback textual

## 11.4 Ejemplo concreto de vecinos

Pregunta:

```text
¿Qué significa SGEIC?
```

Supongamos que el articulo top recuperado es el 41.

Entonces:

- se recupera el articulo 41 porque define el concepto
- se recuperan articulos vecinos 39, 40, 42 y 43
- si el chunk top de `art_41` es `art_41_chunk_001`, tambien se recuperan:
  - `art_41_chunk_000`
  - `art_41_chunk_002`

Esto sirve para capturar:

- definicion
- requisitos adyacentes
- continuidad del articulo

sin tener que meter media ley en contexto.

---

## 12. Como se construye el contexto final

Despues del retrieval, no se pasa todo al LLM.

Se construye un contexto acotado.

## 12.1 Reglas de contexto

Por articulo:

- si el articulo es prioritario, puede entrar completo
- si no, entran solo los chunks mas relevantes

Limites:

- maximo total de caracteres de contexto
- maximo de chunks por articulo

## 12.2 Como se eligen los chunks finales

La funcion de seleccion de chunks hace esto:

1. calcula score por query para cada chunk relevante
2. toma el chunk semilla mas fuerte
3. antes de meter otros chunks lejanos, intenta meter chunks vecinos del semilla
4. rellena hasta el limite

Eso hace que el contexto final tenga mas continuidad local.

Ejemplo:

Si los chunks relevantes de un articulo son:

- chunk 0
- chunk 3
- chunk 4

y el chunk 3 es el mejor para la query, el sistema preferira:

- chunk 3
- chunk 4

antes que:

- chunk 3
- chunk 0

porque 4 es vecino local del chunk semilla.

---

## 13. Generacion de respuesta

Una vez construido el contexto:

1. se invoca el LLM
2. se le pasa la pregunta original
3. se le pasa el contexto ya recortado
4. genera una respuesta corta y juridicamente aterrizada

El retrieval puede usar query expandida.

La generacion usa la pregunta original.

Eso es importante porque:

- mejora recall en retrieval
- evita que la respuesta final parezca escrita sobre una pregunta artificialmente expandida

---

## 14. Validaciones posteriores a la generacion

Despues de generar, el sistema aun no da la respuesta por buena automaticamente.

Hace dos controles:

## 14.1 Refuerzo de citas

Si el usuario pidio un articulo concreto y la respuesta no cita ese articulo:

1. se vuelve a recuperar el articulo directo
2. se recompone contexto
3. se reintenta la sintesis

## 14.2 Groundedness check

El sistema pregunta al LLM, en esencia:

> "¿La respuesta esta completamente soportada por el contexto?"

Si la respuesta no esta soportada, el sistema responde:

```text
NO ENCONTRADO EN EL CONTEXTO
```

Esto reduce alucinaciones.

---

## 15. Ejemplo completo paso a paso

Vamos a seguir una pregunta real:

```text
¿Qué significa SGEIC?
```

Asumimos modo `grafo`.

### Paso 1. Entrada del usuario

El usuario lanza:

```text
¿Qué significa SGEIC?
```

### Paso 2. Clasificacion

La pregunta se clasifica como definicional.

### Paso 3. Expansion de query

Se expande aproximadamente a:

```text
¿Qué significa SGEIC? sociedad gestora sociedades gestoras concepto
```

### Paso 4. Seeds juridicas

La presencia de `SGEIC` hace que se siembre el articulo 41.

### Paso 5. Fulltext y recuperacion inicial

Se buscan:

- chunks con `SGEIC`
- articulos con `SGEIC`
- titulos tipo `Concepto`, `Definición`, `Régimen jurídico`

En este caso lo esperable es que aparezcan:

- art. 41 como principal
- art. 42 cerca si la consulta roza requisitos
- otros articulos cercanos o con menciones de SGEIC

### Paso 6. Expansion por subgrafo

Si los articulos top tienen referencias o comparten seccion/capitulo, se añaden candidatos.

### Paso 7. Reranking

El sistema ordena los articulos por:

- relevancia lexical
- score previo
- señales estructurales
- prioridad de semillas

### Paso 8. Vecinos locales

Como el articulo 41 es top:

- se añaden articulos 39, 40, 42 y 43 como vecinos numericos
- si el chunk mas fuerte de `art_41` es el `chunk_001`, se añaden tambien `chunk_000` y `chunk_002`

### Paso 9. Construccion de contexto

El sistema decide:

- si mete el articulo 41 completo
- o si mete solo sus fragmentos top
- y hace lo mismo con el resto de articulos hasta alcanzar el maximo de contexto

### Paso 10. Generacion

Se manda al LLM:

- la pregunta original: `¿Qué significa SGEIC?`
- el contexto final recuperado

### Paso 11. Validacion

Se comprueba:

- si la respuesta cita los articulos necesarios
- si esta soportada por el contexto

### Paso 12. Salida final

La salida ideal seria algo del estilo:

```text
Artículos relevantes: artículo 41.
Respuesta: Las SGEIC son sociedades anónimas o de responsabilidad limitada cuyo objeto social es la gestión de las inversiones de una o varias ECR y EICC, así como el control y gestión de sus riesgos.
```

---

## 16. Ejemplo de cuando se buscan vecinos y cuando no

## Caso A: si se buscan vecinos

Pregunta:

```text
¿Qué diferencia hay entre una SCR y un FCR?
```

Flujo esperado:

- seeds: art. 26 y art. 30
- retrieval definicional
- vecinos locales alrededor de 26 y 30
- chunks vecinos en los articulos top si estan troceados

Se buscan vecinos porque:

- hay articulos semilla claros
- la consulta es comparativa
- el contexto adyacente puede completar requisitos o regimen

## Caso B: no se buscan vecinos de todos los resultados

Supongamos que se recuperan 12 articulos candidatos.

El sistema no hace `±2` para los 12.

Lo hace solo para los top articulos tras reranking.

Eso evita una explosion combinatoria del contexto.

## Caso C: modo cypher puro con filas suficientes

Si el modo `cypher` obtiene filas utiles y construye contexto a partir de esas filas, no hay una expansion local completa sobre todas esas filas como si fueran articulos top del modo `grafo`.

La vecindad local entra sobre:

- articulos directos
- fallback textual
- retrieval basado en articulos/chunks

---

## 17. Resumen ejecutivo final

El sistema actual puede resumirse asi:

### A. Estructura del conocimiento

La ley se representa como:

- articulos completos
- chunks estructurados por articulo
- jerarquia legal
- referencias cruzadas
- entidades y conceptos

### B. Chunking

El chunking es:

- estructurado
- orientado a articulos
- con contexto jerarquico
- con overlap

### C. Retrieval

El retrieval es:

- hibrido
- chunk-first para recuperar
- article-level para rerankear y responder
- reforzado con expansion de query
- reforzado con vecinos locales

### D. Generacion

La generacion usa:

- pregunta original
- contexto recortado
- validacion de citas
- groundedness check

### E. Filosofia tecnica actual

La filosofia del sistema no es:

> "buscar unos trozos y dejarselos al LLM"

La filosofia actual es mas bien:

> "recuperar fragmentos utiles, reagruparlos juridicamente por articulo, expandir lo justo por grafo y vecindad local, y solo entonces sintetizar una respuesta controlada"

---

## 18. Punto de mejora natural

Si en el futuro quieres seguir refinandolo, los siguientes pasos tecnicos mas naturales serian:

1. mover algunos parametros de retrieval a `config.py`
2. loggear en disco el contexto final realmente enviado al LLM
3. separar mas claramente:
   - retrieval candidate generation
   - reranking
   - context assembly
4. añadir evaluacion especifica de recall por articulo esperado

---

## 19. Principal problema actual detectado

El principal problema tecnico que veo en el RAG actual no es ya la falta de recall, sino el riesgo contrario:

## sobreexpansion del contexto y perdida de precision juridica

Dicho de forma simple:

el sistema esta bastante optimizado para no dejar fuera articulos potencialmente utiles, pero eso hace que en algunos casos recupere demasiado contexto adyacente o estructuralmente cercano.

### 19.1 Por que considero que este es el problema principal

Hoy el pipeline acumula varias capas orientadas a recall:

1. expansion de query
2. semillas juridicas por siglas o terminos
3. fulltext sobre chunks
4. fulltext sobre articulos
5. expansion por subgrafo:
   - referencias
   - misma seccion
   - mismo capitulo
6. vecinos locales:
   - articulos `+-2`
   - chunks `+-2`

Cada una de estas capas, por separado, tiene sentido.

El problema es que, sumadas, pueden meter en el contexto final articulos que:

- son cercanos estructuralmente
- son semanticamente proximos
- pero no son estrictamente necesarios para responder la pregunta concreta

En legal RAG eso es delicado, porque el LLM puede mezclar:

- definicion
- requisitos
- regimen juridico
- limites
- obligaciones

aunque el usuario solo estuviera preguntando por una de esas cosas.

### 19.2 Donde se ve este riesgo en la arquitectura actual

El problema no suele estar en una sola funcion aislada, sino en la combinacion de estas capas:

- `buscar_articulos(...)`
- `_expandir_subgrafo(...)`
- `_expandir_vecindad_local(...)`
- `construir_contexto(...)`

La idea general es buena:

- recuperar primero
- rerankear despues
- recortar contexto al final

Pero la recuperacion ya llega bastante inflada al punto de ensamblado del contexto.

### 19.3 Ejemplo real del riesgo

Pregunta:

```text
¿Qué significa SGEIC?
```

Lo ideal juridicamente seria contestar casi enteramente con el articulo 41.

Pero el sistema puede recuperar, ademas:

- articulo 41 como definicion principal
- articulos 39 y 40 por vecindad numerica
- articulo 42 por vecindad numerica
- articulo 43 por vecindad numerica
- articulos adicionales por misma seccion o mismo capitulo

Desde el punto de vista del recall, esto esta bien.

Desde el punto de vista de precision juridica, ya no tanto:

- el 41 define el concepto
- el 42 entra ya en requisitos de acceso
- el 39 y 40 regulan otras piezas del sistema

Resultado posible:

el modelo responde con algo apoyado en el contexto, pero mezcla definicion y requisitos o contexto normativo cercano.

Eso es importante:

el groundedness check no detecta este problema si la respuesta esta efectivamente soportada por el contexto recuperado.

Es decir:

- la respuesta puede estar "grounded"
- y aun asi ser menos precisa de lo deseable para la pregunta concreta

### 19.4 Por que este problema es mas grave que una simple alucinacion

Una alucinacion pura se puede cortar con:

- prompt estricto
- fallback `NO ENCONTRADO EN EL CONTEXTO`
- groundedness check

Pero la contaminacion de contexto es mas dificil de detectar porque:

1. el modelo no esta inventando
2. la informacion si esta en el contexto
3. el problema es que el contexto no era el mejor contexto

En otras palabras:

el sistema puede equivocarse "con fundamento", porque el fundamento recuperado no era el minimo ni el mas preciso.

### 19.5 Manifestacion tipica en preguntas legales

Este problema aparece sobre todo en preguntas de tipo:

- definicion:
  - `¿Qué significa SGEIC?`
  - `¿Qué es una SCR?`
- comparativa:
  - `¿Qué diferencia hay entre una SCR y un FCR?`
- preguntas cortas y cerradas:
  - `¿Puede una ECR invertir en empresas cotizadas?`

En ese tipo de preguntas, meter mucho contexto cercano puede empeorar la respuesta porque el modelo tiene mas material del necesario para mezclar.

### 19.6 Relacion con los vecinos locales

La recuperacion de vecinos `+-2` tiene una ventaja clara:

- evita perder continuidad inmediata del articulo o de la zona normativa

Pero tambien introduce una asuncion fuerte:

- que la cercania numerica implica relevancia para la pregunta

Esa asuncion es util muchas veces, pero no siempre es juridicamente cierta.

Dos articulos consecutivos pueden tratar:

- definicion en uno
- requisitos operativos en el siguiente
- infracciones un poco mas adelante

y no todos deben entrar en una respuesta definicional.

### 19.7 Formulacion corta del problema principal

Si tuviera que resumirlo en una sola frase, seria esta:

> el RAG actual esta mas cerca de sobrerrecuperar contexto relevante que de quedarse corto, y ese exceso de contexto puede degradar la precision juridica aunque la respuesta siga estando soportada por el contexto

### 19.8 La consecuencia practica

La consecuencia mas importante no es que el sistema "falle a lo bruto", sino que a veces responda con:

- demasiado contexto
- mezcla de articulos adyacentes
- inclusion de matices no pedidos
- respuestas menos limpias de lo que exige una consulta juridica cerrada

### 19.9 La linea de mejora mas importante

Si solo hubiera que priorizar una mejora tecnica a partir del estado actual, yo priorizaria:

## hacer mas selectiva la expansion posterior al retrieval

En la practica, eso significaria algo como:

1. usar vecinos solo cuando el score del articulo semilla sea alto pero el contexto sea incompleto
2. usar vecinos solo en preguntas comparativas o de condiciones, no tanto en definiciones puras
3. introducir una fase de pruning final por tipo de pregunta antes de construir el contexto
4. penalizar articulos vecinos que no compartan senales lexicas fuertes con la pregunta

Esa mejora atacaria el punto que hoy veo como mas delicado en el sistema.
