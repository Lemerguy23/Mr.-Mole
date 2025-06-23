<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${self.title()}</title>
    <link rel="icon" href="/static/logo/logo.png" type="image/png">
    <link rel="stylesheet" href="/static/css_content/all_styles.css">
</head>
<body>
    <header class="header">
        <div class="logo-container">
        <img src="/static/logo/logo.png" alt="Mr.Mole Logo" class="logo-img">
        <span class="logo-text">Mr.Mole</span>
        </div>
        <nav class="nav-links">
            <a href="/" id="home-link" class="nav-link">Главная</a>
            <a href="/analyze" id="analyze-link" class="nav-link">Анализ</a>
            <a href="/faq" id="faq-link" class="nav-link">FAQ</a>
            <a href="static/app/app-arm64-v8a-release.apk" 
               id="app-link" 
               class="nav-link cta-button"
               download="Mr. Mole.apk">Скачать приложение</a>
        </nav>
    </header>
    <main class="main-content">
        ${self.content()}
    </main>
    <footer class="footer">
        <div class="container">
            <p>© 2025 Mr.Mole. Не все права защищены...</p>
        </div>
    </footer>
    ${self.scripts()}
</body>
</html>

<%def name="title()">
Mr.Mole - ${self.page_title()}
</%def>

<%def name="page_title()">
Анализ родинок
</%def>

<%def name="scripts()">
    <script src="/static/scripts/nav.js"></script>
</%def>