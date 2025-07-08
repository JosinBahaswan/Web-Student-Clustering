@echo off
echo ========================================
echo    INSTALLING PYTHON REQUIREMENTS
echo ========================================
echo.

echo Checking if Python is installed...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python is installed. Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo.
echo Installing required packages...
echo.

echo Installing streamlit...
pip install streamlit

echo Installing pandas...
pip install pandas

echo Installing numpy...
pip install numpy

echo Installing scikit-learn...
pip install scikit-learn

echo Installing plotly...
pip install plotly

echo.
echo ========================================
echo    INSTALLATION COMPLETE!
echo ========================================
echo.
echo All required packages have been installed.
echo You can now run the app using: run_app.bat
echo.
pause 