# TrintonLexer — General Triton Lexer

Lexer para **Mini-Triton**, un DSL simplificado inspirado en OpenAI Triton.  
Desarrollado como componente de la tarea de compiladores.

## Estructura

```
TrintonLexer/
├── lexer.py          ← tokenizador (módulo importable + CLI)
├── tests/
│   ├── test_lexer.py ← suite pytest (~55 tests)
│   ├── valid_01.tri  ← casos léxicamente válidos
│   │   ...
│   ├── valid_07.tri
│   ├── invalid_01.tri ← casos con errores sintácticos (válidos léxicamente)
│   │   ...
│   └── invalid_06.tri
└── README.md
```

## Uso como CLI

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

## Uso como módulo

```python
from lexer import tokenize, Token, ID, NUMBER, AT, DOT

tokens = tokenize("@triton.jit def add(x, y): { z = x + y; }")

for tok in tokens:
    print(tok)
# Token(AT         '@'             línea=1 col=1)
# Token(ID         'triton'        línea=1 col=2)
# ...
```

## Tipos de token

| Tipo       | Patrón / carácter             |
|------------|-------------------------------|
| `AT`       | `@`                           |
| `COLON`    | `:`                           |
| `COMMA`    | `,`                           |
| `DOT`      | `.`                           |
| `EQUALS`   | `=`                           |
| `LBRACE`   | `{`                           |
| `LPAREN`   | `(`                           |
| `MINUS`    | `-`                           |
| `NUMBER`   | `\d+(\.\d+)?`                 |
| `PLUS`     | `+`                           |
| `RBRACE`   | `}`                           |
| `RPAREN`   | `)`                           |
| `SEMICOLON`| `;`                           |
| `SLASH`    | `/`                           |
| `STAR`     | `*`                           |
| `ID`       | `[A-Za-z_]\w*`                |
| `EOF`      | token sintético al final      |

> **`tl.load`** se produce como `ID("tl") DOT ID("load")`.  
> Palabras clave como `def`, `jit`, `triton` se tokenan como `ID` y el parser las valida por valor.  
> Comentarios de línea (`# ...`) y espacios son descartados silenciosamente.

## Correr los tests

```bash
pip install pytest
pytest tests/test_lexer.py -v
```

## Requisitos

- Python 3.9+
- Sin dependencias externas (solo stdlib)
