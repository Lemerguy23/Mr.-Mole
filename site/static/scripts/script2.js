document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const resultContainer = document.getElementById('resultContainer');
    const resultValue = document.getElementById('resultValue');
    const resultDescription = document.getElementById('resultDescription');
    const errorContainer = document.getElementById('errorContainer');
    const submitBtn = uploadForm.querySelector('button[type="submit"]');

    // Обработка выбора файла
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            fileNameDisplay.textContent = this.files[0].name;
            fileNameDisplay.classList.add('active');
        } else {
            fileNameDisplay.textContent = '';
            fileNameDisplay.classList.remove('active');
        }
    });

    // Отправка формы
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Сброс состояния
        errorContainer.style.display = 'none';
        resultContainer.style.display = 'none';
        submitBtn.disabled = true;
        submitBtn.textContent = 'Анализируем...';

        if (!fileInput.files.length) {
            showError('Пожалуйста, выберите изображение');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Проанализировать';
            return;
        }

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const response = await fetch('/check-mole', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Ошибка сервера');
            }

            const data = await response.json();
            displayResult(data.class);

        } catch (error) {
            showError(error.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Проанализировать';
        }
    });

    function displayResult(classId) {
        resultValue.textContent = classId;
        
        if (classId == 1) {
            resultValue.className = 'result-value malignant';
            resultDescription.textContent = 'Обнаружены признаки злокачественности';
        } else {
            resultValue.className = 'result-value benign';
            resultDescription.textContent = 'Доброкачественное образование';
        }
        
        resultContainer.style.display = 'block';
    }

    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';
    }
});