<%inherit file="base.mako"/>

<%def name="title()">
    Анализ родинок
</%def>

<%def name="content()">
    <h1>Анализ родинок на злокачественность</h1>
    
    <section class="upload-section">
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="fileInput" class="file-label">
                    <span>Выберите изображение родинки</span>
                    <input type="file" id="fileInput" name="file" accept="image/*" required>
                </label>
                <div id="fileNameDisplay" class="file-name-display"></div>
            </div>
            
            <div id="previewContainer" style="display: none;">
                <div class="preview-wrapper">
                    <img id="imagePreview" style="display: block; max-width: 100%; max-height: 60vh;">
                </div>
            </div>
            
            <div class="selection-buttons" style="margin-top: 15px;">
                <button type="button" id="selectAreaBtn" class="submit-btn">Выделить область</button>
                <button type="button" id="confirmSelectionBtn" class="submit-btn confirm-btn" style="display: none;">Подтвердить</button>
                <button type="button" id="cancelSelectionBtn" class="submit-btn cancel-btn" style="display: none;">Отменить</button>
            </div>
            
            <button type="submit" id="analyzeBtn" class="submit-btn">Проанализировать</button>
        </form>
    </section>

    <section id="resultContainer" class="result-section" style="display: none;">
        <h2>Результат анализа</h2>
        <div class="result-box">
            <div class="result-value" id="resultValue"></div>
            <div class="result-description" id="resultDescription"></div>
        </div>
    </section>

    <div id="errorContainer" class="error-message" style="display: none;"></div>
</%def>

<%def name="scripts()">
    ${parent.scripts()}
    <script src="/static/scripts/analyze.js"></script>
</%def>