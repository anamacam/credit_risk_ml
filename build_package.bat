@echo off
set "VENV_NAME=.venv"
set "TEMP_MOVE=..\%VENV_NAME%_temp"

echo [1/4] Protegiendo el entorno virtual...
if exist %VENV_NAME% (
    move %VENV_NAME% %TEMP_MOVE%
)

echo [2/4] Limpiando caches de Conda...
call conda build purge-all

echo [3/4] Construyendo paquete profesional...
:: Usamos "recipe" porque movimos el meta.yaml alli
call conda-build recipe

echo [4/4] Restaurando el entorno virtual...
if exist %TEMP_MOVE% (
    move %TEMP_MOVE% %VENV_NAME%
)

echo === PROCESO FINALIZADO CON EXITO ===
pause