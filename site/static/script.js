document.addEventListener('DOMContentLoaded', () => {
    // Основные элементы
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const resultContainer = document.getElementById('resultContainer');
    const processedImage = document.getElementById('processedImage');
    const downloadBtn = document.getElementById('downloadBtn');
    const errorContainer = document.getElementById('errorContainer');
    const fileInfo = document.createElement('div'); // Блок для информации о файле

    // Добавляем блок информации в форму
    fileInfo.className = 'file-info';
    fileInput.parentNode.insertBefore(fileInfo, fileInput.nextSibling);

    // Обработчик выбора файла
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
            fileInfo.textContent = `Выбран файл: ${file.name} (${fileSizeMB} MB)`;
            
            // Проверка размера (максимум 10MB)
            if (file.size > 10 * 1024 * 1024) {
                fileInfo.textContent = 'Файл слишком большой (макс. 10MB)';
                fileInput.value = '';
            }
        } else {
            fileInfo.textContent = '';
        }
    });

    // Обработчик отправки формы
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            errorContainer.textContent = 'Пожалуйста, выберите файл';
            errorContainer.style.display = 'block';
            return;
        }

        // Сброс состояния
        errorContainer.style.display = 'none';
        resultContainer.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                processedImage.src = data.image_url;
                downloadBtn.href = data.image_url;
                resultContainer.style.display = 'block';
                fileInfo.textContent = ''; // Очищаем информацию о файле
            } else {
                errorContainer.textContent = data.message || 'Ошибка при обработке';
                errorContainer.style.display = 'block';
            }
        } catch (error) {
            errorContainer.textContent = 'Ошибка соединения';
            errorContainer.style.display = 'block';
            console.error('Error:', error);
        }
    });
});
