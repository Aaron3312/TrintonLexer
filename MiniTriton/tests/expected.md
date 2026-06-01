# Resultados esperados — Mini-Triton Parser

| Archivo        | Resultado | Razón / Mensaje del parser                                                         |
|----------------|-----------|------------------------------------------------------------------------------------|
| valid_01.tri   | VALIDO    | Asignación simple `z = x + y` — Kernel(add, params=[x, y])                        |
| valid_02.tri   | VALIDO    | Paréntesis y precedencia: `(x+1)*2` → BinaryOp("*") sobre BinaryOp("+")           |
| valid_03.tri   | VALIDO    | Tres parámetros; `b*c` se agrupa antes de sumar con `a`                            |
| valid_04.tri   | VALIDO    | ExprStmt con llamada calificada `tl.load(x)`                                       |
| valid_05.tri   | VALIDO    | Call("foo") con tres args: Name, Number, BinaryOp("+")                             |
| valid_06.tri   | VALIDO    | Parámetro anotado `BS: tl.constexpr` + llamada `tl.arange`                        |
| valid_07.tri   | VALIDO    | Kernel B completo: 3 stmts, llamada anidada `tl.store(..., tl.load(...))`          |
| invalid_01.tri | INVALIDO  | `se esperaba '@triton.jit' pero llegó 'def' en línea 1, col 1`                    |
| invalid_02.tri | INVALIDO  | `se esperaba '{' pero llegó 'y' en línea 2, col 19`                               |
| invalid_03.tri | INVALIDO  | `se esperaba ';' pero llegó '}' en línea 4, col 1`                                |
| invalid_04.tri | INVALIDO  | `se esperaba un operando pero llegó ';' en línea 3, col 11`                       |
| invalid_05.tri | INVALIDO  | `argumento con nombre 'mask=...' no permitido en línea 3, col 18`                 |
| invalid_06.tri | INVALIDO  | `'if' no está soportado en Mini-Triton en línea 3, col 3`                         |
