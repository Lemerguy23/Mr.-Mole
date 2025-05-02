<%inherit file="base.mako"/>

<%def name="title()">
Image Processor - Главная
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
            <button type="submit" class="submit-btn">Проанализировать</button>
        </form>
    </section>

    <section id="resultContainer" class="result-section">
        <h2>Результат анализа</h2>
        <div class="result-box">
            <div class="result-value" id="resultValue"></div>
            <div class="result-description" id="resultDescription"></div>
        </div>
    </section>

    <div id="errorContainer" class="error-message"></div>
</%def>

<%def name="scripts()">
    ${parent.scripts()}
    <script src="/static/scripts/script2.js"></script>
</%def>