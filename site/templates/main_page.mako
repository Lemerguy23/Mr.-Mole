<%inherit file="base.mako"/>

<%def name="title()">
    Mr. Mole - Главная
</%def>

<%def name="content()">
    <div class="main-page-container">
        <section class="hero-section">
            <h1>Ранняя диагностика меланомы</h1>
            <p class="subtitle">Искусственный интеллект для анализа родинок и кожных образований</p>
            
            <div class="cta-buttons">
                <a href="/analyze" class="cta-button primary">Начать анализ</a>
                <a href="#" class="cta-button secondary">Мобильное приложение (скоро)</a>
            </div>
        </section>

        <section class="info-section">
            <div class="info-card">
                <h2>Как это работает?</h2>
                <p>Наша нейросеть анализирует фотографии кожных образований и оценивает вероятность злокачественности с точностью 92%.</p>
            </div>

            <div class="info-card">
                <h2>Почему это важно?</h2>
                <p>Ранняя диагностика меланомы увеличивает шансы успешного лечения до 99%. Не откладывайте проверку!</p>
            </div>
        </section>
    </div>
</%def>
