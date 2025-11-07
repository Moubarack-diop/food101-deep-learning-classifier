@echo off
REM Script de compilation du rapport LaTeX
REM Necessite une installation LaTeX (MiKTeX ou TeX Live)

echo ========================================
echo Compilation du Rapport LaTeX
echo ========================================
echo.

REM Verifier si pdflatex est installe
where pdflatex >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: pdflatex n'est pas installe ou pas dans le PATH
    echo.
    echo Telechargez et installez MiKTeX depuis:
    echo https://miktex.org/download
    echo.
    pause
    exit /b 1
)

echo [1/4] Premiere compilation...
pdflatex -interaction=nonstopmode rapport_projet.tex
if %errorlevel% neq 0 (
    echo ERREUR lors de la premiere compilation
    pause
    exit /b 1
)

echo.
echo [2/4] Deuxieme compilation (pour les references)...
pdflatex -interaction=nonstopmode rapport_projet.tex

echo.
echo [3/4] Troisieme compilation (pour la table des matieres)...
pdflatex -interaction=nonstopmode rapport_projet.tex

echo.
echo [4/4] Nettoyage des fichiers temporaires...
del rapport_projet.aux 2>nul
del rapport_projet.log 2>nul
del rapport_projet.out 2>nul
del rapport_projet.toc 2>nul
del rapport_projet.lof 2>nul
del rapport_projet.lot 2>nul

echo.
echo ========================================
echo Compilation terminee avec succes!
echo Fichier genere: rapport_projet.pdf
echo ========================================
echo.

REM Ouvrir le PDF automatiquement
if exist rapport_projet.pdf (
    echo Ouverture du PDF...
    start rapport_projet.pdf
)

pause
