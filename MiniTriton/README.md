# Mini-Triton — Lexer + Parser

Analizador léxico y sintáctico para **Mini-Triton**, un DSL simplificado inspirado en OpenAI Triton.
Desarrollado como entregable de la actividad **3.2 — Gramáticas libres de contexto** del curso de Compiladores, Tecnológico de Monterrey.

**Integrantes — Grupo 504**

| Nombre | Matrícula |
|---|---|
| Alexei Delgado De Gante | A01637405 |
| Luis Fernando Cuevas Arroyo | A01647254 |
| Aaron Hernandez Jimenez | A01642529 |
| Sebastian Cervera Maltos | A01068436 |

---

## Estructura del proyecto

```
MiniTriton/
├── lexer.py                  ← Analizador léxico (módulo importable + CLI)
├── mini_triton_parser.py     ← Analizador sintáctico + construcción de AST (CLI principal)
├── tests/
│   ├── test_lexer.py         ← Suite pytest del lexer (~55 tests)
│   ├── valid_01.tri          ← Casos léxica y sintácticamente válidos
│   ├── valid_02.tri
│   ├── valid_03.tri
│   ├── valid_04.tri
│   ├── valid_05.tri
│   ├── valid_06.tri
│   ├── valid_07.tri
│   ├── invalid_01.tri        ← Casos con errores sintácticos
│   ├── invalid_02.tri
│   ├── invalid_03.tri
│   ├── invalid_04.tri
│   ├── invalid_05.tri
│   └── invalid_06.tri
└── README.md
```

---

## Requisitos

- **Python 3.9+**
- Sin dependencias externas (solo stdlib)
- Para correr la suite del lexer: `pip install pytest`

---

## Componente 1 — Lexer (`lexer.py`)

Tokenizador que escanea un archivo fuente `.tri` y produce una lista de tokens tipados.

### Uso como CLI

```bash
python lexer.py tests/valid_01.tri
```

Salida de ejemplo:

```
#     TIPO         VALOR                LÍNEA    COL
-------------------------------------------------------
0     AT           '@'                  1        1
1     ID           'triton'             1        2
2     DOT          '.'                  1        8
3     ID           'jit'                1        9
4     ID           'def'                1        13
5     ID           'add'                1        17
...
```

> En Windows, si hay problemas con caracteres Unicode: `python -X utf8 lexer.py <archivo>`

### Uso como módulo

```python
from lexer import tokenize, Token, ID, NUMBER, AT, DOT

tokens = tokenize("@triton.jit def add(x, y): { z = x + y; }")

for tok in tokens:
    print(tok)
# Token(AT         '@'             línea=1 col=1)
# Token(ID         'triton'        línea=1 col=2)
# ...
```

### Tipos de token producidos (17)

| Tipo | Patrón / carácter |
|---|---|
| `AT` | `@` |
| `COLON` | `:` |
| `COMMA` | `,` |
| `DOT` | `.` |
| `EQUALS` | `=` |
| `LBRACE` | `{` |
| `LPAREN` | `(` |
| `MINUS` | `-` |
| `NUMBER` | `\d+(\.\d+)?` |
| `PLUS` | `+` |
| `RBRACE` | `}` |
| `RPAREN` | `)` |
| `SEMICOLON` | `;` |
| `SLASH` | `/` |
| `STAR` | `*` |
| `ID` | `[A-Za-z_]\w*` |
| `EOF` | token sintético al final |

**Supuestos del lexer:**
- `tl.load` se produce como `ID("tl") DOT ID("load")` — el parser los reensambla.
- Palabras clave como `def`, `jit`, `triton` se tokenizan como `ID`; el parser las valida por valor.
- Comentarios de línea (`# ...`) y espacios en blanco se descartan silenciosamente.
- Lanza `SyntaxError` si encuentra un carácter no reconocido por la gramática.

### Correr los tests del lexer

```bash
pytest tests/test_lexer.py -v
```

---

## Componente 2 — Parser (`mini_triton_parser.py`)

Parser descendente recursivo LL(1) que consume los tokens del lexer, construye un AST tipado y lo serializa en texto.

### Uso como CLI

```bash
# Archivo válido → imprime VALIDO + AST
python mini_triton_parser.py tests/valid_07.tri

# Archivo inválido → imprime INVALIDO + mensaje de error + posición
python mini_triton_parser.py tests/invalid_05.tri

# Windows (glifos del árbol en UTF-8)
python -X utf8 mini_triton_parser.py <archivo.tri>
```

### Salida — archivo válido (`valid_01.tri`)

```
VALIDO
Program
└── Kernel(name="add", params=[x, y])
    └── Assign
        ├── Name("z")
        └── BinaryOp("+")
            ├── Name("x")
            └── Name("y")
```

### Salida — archivo inválido (`invalid_05.tri`)

```
INVALIDO: argumento con nombre 'mask=...' no permitido en línea 2, col 36
```

### Nodos del AST

| Nodo | Campos | Descripción |
|---|---|---|
| `Program` | `kernel` | Nodo raíz; contiene el único kernel del archivo |
| `Kernel` | `name`, `params`, `body` | Función decorada con `@triton.jit` |
| `Param` | `name`, `annotation` | Parámetro formal; anotación opcional, p. ej. `tl.constexpr` |
| `Assign` | `target`, `value` | Sentencia de asignación `id = expr ;` |
| `ExprStmt` | `expr` | Expresión usada como sentencia `expr ;` |
| `BinaryOp` | `op`, `left`, `right` | Operación binaria `+ - * /` |
| `Call` | `callee`, `args` | Llamada a función; `callee` puede ser `"tl.load"` |
| `Name` | `name`, `line` | Referencia a variable o nombre calificado |
| `Number` | `value`, `line` | Literal entero o decimal |

---

## Supuestos y decisiones de diseño

| Aspecto | Decisión |
|---|---|
| Bloques | Se usan `{ }` en lugar de indentación; elimina tokens INDENT/DEDENT y mantiene la gramática LL(1) |
| Nombres con punto | El lexer produce `ID DOT ID`; `parseNameOrCall()` los reensambla en el string `"tl.load"` |
| Asignación vs. expresión | Lookahead de 2 tokens: si el token actual es `ID` y el siguiente es `=`, se invoca `parseAssign()`; en caso contrario, `parseExprStmt()` |
| Argumentos nombrados | Rechazados explícitamente: `mask=1` dentro de una lista de argumentos lanza `ParseError` |
| Control de flujo | `if`, `while`, `for`, `return` están fuera del alcance; el parser los detecta y reporta un error descriptivo |
| Múltiples kernels | Un archivo contiene exactamente un kernel; fuera del alcance |

---

## Casos de prueba

### Válidos (7)

| Archivo | Descripción |
|---|---|
| `valid_01.tri` | Asignación simple `z = x + y` — Ejemplo A del enunciado |
| `valid_02.tri` | Agrupación con paréntesis + precedencia `* > +` |
| `valid_03.tri` | Tres parámetros; `b*c` se agrupa antes de sumar `a` |
| `valid_04.tri` | `ExprStmt` con llamada calificada `tl.load(x)` |
| `valid_05.tri` | Argumentos mixtos: `Name`, `Number`, `BinaryOp` |
| `valid_06.tri` | Parámetro anotado `BS: tl.constexpr` + llamada `tl.arange` |
| `valid_07.tri` | Ejemplo B completo — llamadas anidadas `tl.store` / `tl.load` |

### Inválidos (6)

| Archivo | Problema esperado |
|---|---|
| `invalid_01.tri` | Falta decorador `@triton.jit` |
| `invalid_02.tri` | Falta delimitador de bloque `{ }` |
| `invalid_03.tri` | Falta `;` al final de sentencia |
| `invalid_04.tri` | Expresión incompleta: `y = x + ;` |
| `invalid_05.tri` | Argumento nombrado: `tl.load(x, mask=1)` |
| `invalid_06.tri` | Flujo de control no soportado: `if (x) { ... }` |