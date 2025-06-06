document.addEventListener('DOMContentLoaded', function () {
    // Основные элементы
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const resultContainer = document.getElementById('resultContainer');
    const resultValue = document.getElementById('resultValue');
    const resultDescription = document.getElementById('resultDescription');
    const errorContainer = document.getElementById('errorContainer');
    const submitBtn = uploadForm.querySelector('button[type="submit"]');

    // Элементы превью
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');

    // Создаем необходимые элементы
    const previewWrapper = document.createElement('div');
    previewWrapper.className = 'preview-wrapper';

    const confirmSelectionBtn = document.createElement('button');
    confirmSelectionBtn.id = 'confirmSelectionBtn';
    confirmSelectionBtn.className = 'submit-btn confirm-btn';
    confirmSelectionBtn.textContent = 'Подтвердить';
    confirmSelectionBtn.style.display = 'none';

    const cancelSelectionBtn = document.createElement('button');
    cancelSelectionBtn.id = 'cancelSelectionBtn';
    cancelSelectionBtn.className = 'submit-btn cancel-btn';
    cancelSelectionBtn.textContent = 'Отменить';
    cancelSelectionBtn.style.display = 'none';

    const buttonsContainer = document.createElement('div');
    buttonsContainer.className = 'selection-buttons';

    // Собираем структуру DOM
    buttonsContainer.appendChild(selectAreaBtn);
    buttonsContainer.appendChild(confirmSelectionBtn);
    buttonsContainer.appendChild(cancelSelectionBtn);

    previewContainer.appendChild(previewWrapper);
    previewWrapper.appendChild(imagePreview);
    previewContainer.appendChild(buttonsContainer);

    // Состояние приложения
    let isSelecting = false;
    let isDragging = false;
    let startX, startY;
    let selectionRect = null;
    let originalImageSrc = null;
    let darkeningOverlay = null;

    // Обработчик выбора файла
    fileInput.addEventListener('change', function (e) {
        if (!e.target.files || !e.target.files[0]) return;

        const file = e.target.files[0];
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.classList.add('active');

        const reader = new FileReader();
        reader.onload = function (e) {
            originalImageSrc = e.target.result;
            imagePreview.onload = function () {
                previewContainer.style.display = 'block';
                errorContainer.style.display = 'none';
                resetSelectionState();

                if (this.naturalWidth < 260 || this.naturalHeight < 260) {
                    showError('Минимальный размер изображения: 260x260 пикселей');
                    return;
                }

                setupImagePreview();
                selectAreaBtn.style.display = 'inline-block';
            };
            imagePreview.src = originalImageSrc;
        };
        reader.readAsDataURL(file);
    });

    function setupImagePreview() {
        // Сбрасываем стили
        imagePreview.style.maxWidth = '100%';
        imagePreview.style.maxHeight = '60vh';
        imagePreview.style.width = '';
        imagePreview.style.height = '';
        removeDarkeningOverlay();

        // Масштабируем большие изображения
        const maxDisplaySize = 600;
        if (imagePreview.naturalWidth > maxDisplaySize || imagePreview.naturalHeight > maxDisplaySize) {
            const ratio = Math.min(
                maxDisplaySize / imagePreview.naturalWidth,
                maxDisplaySize / imagePreview.naturalHeight
            );
            imagePreview.style.width = `${imagePreview.naturalWidth * ratio}px`;
        }
    }

    // Кнопка начала выделения
    selectAreaBtn.addEventListener('click', function () {
        isSelecting = true;
        selectAreaBtn.style.display = 'none';
        confirmSelectionBtn.style.display = 'inline-block';
        cancelSelectionBtn.style.display = 'inline-block';
        createDarkeningOverlay();
    });

    // Кнопка отмены выделения
    cancelSelectionBtn.addEventListener('click', function () {
        resetSelectionState();
        selectAreaBtn.style.display = 'inline-block';
    });

    // Создание затемнения
    function createDarkeningOverlay() {
        darkeningOverlay = document.createElement('div');
        darkeningOverlay.className = 'darkening-overlay';
        previewWrapper.appendChild(darkeningOverlay);
    }

    // Обработчики выделения области
    previewWrapper.addEventListener('mousedown', function (e) {
        if (!isSelecting) return;

        isDragging = true;
        const rect = previewWrapper.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;

        // Создаем прямоугольник выделения
        selectionRect = document.createElement('div');
        selectionRect.className = 'selection-rect';
        selectionRect.style.left = `${startX}px`;
        selectionRect.style.top = `${startY}px`;
        selectionRect.style.width = '0';
        selectionRect.style.height = '0';
        previewWrapper.appendChild(selectionRect);
    });

    previewWrapper.addEventListener('mousemove', function (e) {
        if (!isSelecting || !isDragging || !selectionRect) return;

        const rect = previewWrapper.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        // Обновляем прямоугольник выделения
        const left = Math.min(startX, currentX);
        const top = Math.min(startY, currentY);
        const width = Math.abs(currentX - startX);
        const height = Math.abs(currentY - startY);

        selectionRect.style.left = `${left}px`;
        selectionRect.style.top = `${top}px`;
        selectionRect.style.width = `${width}px`;
        selectionRect.style.height = `${height}px`;

        // Обновляем затемнение
        updateSelectionHighlight({
            left: left,
            top: top,
            right: left + width,
            bottom: top + height
        });
    });

    document.addEventListener('mouseup', function () {
        if (!isSelecting) return;
        isDragging = false;
    });

    function updateSelectionHighlight(rect) {
        if (!darkeningOverlay) return;

        darkeningOverlay.style.clipPath = `polygon(
            0% 0%, 0% 100%,
            ${rect.left}px 100%, ${rect.left}px ${rect.top}px,
            ${rect.right}px ${rect.top}px, ${rect.right}px ${rect.bottom}px,
            ${rect.left}px ${rect.bottom}px, ${rect.left}px 100%,
            100% 100%, 100% 0%
        )`;
    }

    // Кнопка подтверждения выделения
    confirmSelectionBtn.addEventListener('click', function () {
        if (!selectionRect ||
            parseInt(selectionRect.style.width) < 10 ||
            parseInt(selectionRect.style.height) < 10) {
            showError('Выделите область большего размера');
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = 260;
        canvas.height = 260;
        const ctx = canvas.getContext('2d');

        // Масштабируем координаты
        const scaleX = imagePreview.naturalWidth / imagePreview.offsetWidth;
        const scaleY = imagePreview.naturalHeight / imagePreview.offsetHeight;

        const x = parseInt(selectionRect.style.left) * scaleX;
        const y = parseInt(selectionRect.style.top) * scaleY;
        const width = parseInt(selectionRect.style.width) * scaleX;
        const height = parseInt(selectionRect.style.height) * scaleY;

        // Обрезаем и масштабируем изображение
        ctx.drawImage(
            imagePreview,
            x, y, width, height,
            0, 0, 260, 260
        );

        // Обновляем превью
        imagePreview.src = canvas.toDataURL('image/jpeg', 0.9);
        imagePreview.style.width = '260px';
        imagePreview.style.height = '260px';

        resetSelectionState();
        selectAreaBtn.style.display = 'inline-block';
    });

    // Сброс состояния выделения
    function resetSelectionState() {
        if (selectionRect && selectionRect.parentNode) {
            previewWrapper.removeChild(selectionRect);
        }
        if (darkeningOverlay && darkeningOverlay.parentNode) {
            previewWrapper.removeChild(darkeningOverlay);
        }
        selectionRect = null;
        darkeningOverlay = null;
        isSelecting = false;
        isDragging = false;
        confirmSelectionBtn.style.display = 'none';
        cancelSelectionBtn.style.display = 'none';
    }

    function removeDarkeningOverlay() {
        if (darkeningOverlay && darkeningOverlay.parentNode) {
            previewWrapper.removeChild(darkeningOverlay);
            darkeningOverlay = null;
        }
    }

    // Обработка отправки формы
    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();
        processAndSendImage();
    });

    // Замените функцию processAndSendImage на эту:
    async function processAndSendImage() {
        errorContainer.style.display = 'none';
        resultContainer.style.display = 'none';
        setButtonsDisabled(true);

        try {
            const formData = new FormData();
            let imageBlob;

            if (isSelecting && selectionRect) {
                // Создаем canvas с выделенной областью
                const canvas = document.createElement('canvas');
                canvas.width = 260;
                canvas.height = 260;
                const ctx = canvas.getContext('2d');

                // Рассчитываем масштаб
                const scaleX = imagePreview.naturalWidth / imagePreview.offsetWidth;
                const scaleY = imagePreview.naturalHeight / imagePreview.offsetHeight;

                const x = parseInt(selectionRect.style.left) * scaleX;
                const y = parseInt(selectionRect.style.top) * scaleY;
                const width = parseInt(selectionRect.style.width) * scaleX;
                const height = parseInt(selectionRect.style.height) * scaleY;

                // Рисуем выделенную область
                ctx.drawImage(
                    imagePreview,
                    x, y, width, height,
                    0, 0, 260, 260
                );

                // Конвертируем в Blob
                imageBlob = await new Promise(resolve => {
                    canvas.toBlob(resolve, 'image/jpeg', 0.9);
                });
            } else {
                // Используем оригинальное изображение
                const response = await fetch(imagePreview.src);
                imageBlob = await response.blob();
            }

            formData.append('file', imageBlob, 'analysis_image.jpg');

            const response = await fetch('/check-mole', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Ошибка сервера');
            }

            const data = await response.json();
            displayResult(data.class, data.confidence);

        } catch (error) {
            showError(error.message);
        } finally {
            setButtonsDisabled(false);
        }
    }
    function displayResult(classId, confidence) {
        resultValue.textContent = `${classId} (${(confidence * 100).toFixed(1)}%)`;

        switch (classId) {
            case 2:
                resultValue.className = 'result-value malignant';
                resultDescription.textContent = 'Высокая вероятность злокачественного образования';
                break;
            case 1:
                resultValue.className = 'result-value suspicious';
                resultDescription.textContent = 'Признаки подозрительные, требуется дополнительное обследование';
                break;
            case 0:
                resultValue.className = 'result-value benign';
                resultDescription.textContent = 'Доброкачественное образование';
                break;
            default:
                resultValue.className = 'result-value unknown';
                resultDescription.textContent = 'Неизвестный результат диагностики';
        }

        resultContainer.style.display = 'block';
    }

    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';
    }

    function setButtonsDisabled(disabled) {
        submitBtn.disabled = disabled;
        selectAreaBtn.disabled = disabled;
        confirmSelectionBtn.disabled = disabled;
        cancelSelectionBtn.disabled = disabled;

        if (!disabled) {
            submitBtn.textContent = 'Проанализировать';
            //selectAreaBtn.textContent = 'Выделить область';
            confirmSelectionBtn.textContent = 'Подтвердить';
            cancelSelectionBtn.textContent = 'Отменить';
        }
    }
});