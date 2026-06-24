/**
 * CinemaScope — 聊天交互脚本
 * 发送消息、渲染电影卡片、点击跳转详情
 */
let chatHistory = [];

function sendExample(text) {
    document.getElementById('chatInput').value = text;
    sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');
    const typing = document.getElementById('chatTyping');
    const examples = document.getElementById('chatExamples');
    const message = input.value.trim();

    if (!message) return;

    // 隐藏示例按钮
    if (examples) examples.style.display = 'none';

    // 显示用户消息
    appendMessage('user', message);
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    // 显示输入指示器
    typing.classList.add('chat-typing--visible');
    scrollToBottom();

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
            }),
        });

        typing.classList.remove('chat-typing--visible');

        if (!resp.ok) {
            appendMessage('assistant', '抱歉，出了点问题。请稍后再试。');
            return;
        }

        const data = await resp.json();

        // 更新历史
        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: data.reply });

        // 渲染助手回复
        appendAssistantResponse(data);
    } catch (e) {
        typing.classList.remove('chat-typing--visible');
        appendMessage('assistant', '抱歉，连接失败。请确认服务器正在运行。\n\n你可以通过命令行启动：\n`python main.py serve`');
    } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

function appendMessage(role, text) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `chat-bubble chat-bubble--${role}`;

    // 将换行转为 <br>，将 **text** 转为 <strong>
    const formatted = text
        .replace(/\*\*(.+?)\*\*/g, '<strong style="color: var(--gold-bright);">$1</strong>')
        .replace(/\n/g, '<br>');
    div.innerHTML = formatted;

    container.appendChild(div);
    scrollToBottom();
}

function appendAssistantResponse(data) {
    const container = document.getElementById('chatMessages');
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-bubble chat-bubble--assistant';

    // 文本回复
    if (data.reply) {
        const textDiv = document.createElement('div');
        const formatted = data.reply
            .replace(/\*\*(.+?)\*\*/g, '<strong style="color: var(--gold-bright);">$1</strong>')
            .replace(/\n/g, '<br>');
        textDiv.innerHTML = formatted;
        wrapper.appendChild(textDiv);
    }

    // 电影卡片
    if (data.has_cards && data.cards.length > 0) {
        const cardsGrid = document.createElement('div');
        cardsGrid.className = 'chat-cards';

        data.cards.forEach(card => {
            const cardEl = createMovieCardElement(card);
            cardsGrid.appendChild(cardEl);
        });

        wrapper.appendChild(cardsGrid);
    }

    container.appendChild(wrapper);
    scrollToBottom();
}

function createMovieCardElement(card) {
    const a = document.createElement('a');
    a.href = `/movie/${card.movie_idx}`;
    a.className = 'movie-card';

    const posterWrap = document.createElement('div');
    posterWrap.className = 'movie-card__poster-wrap';

    if (card.poster_url) {
        const img = document.createElement('img');
        img.className = 'movie-card__poster';
        img.src = card.poster_url;
        img.alt = card.title;
        img.loading = 'lazy';
        posterWrap.appendChild(img);
    } else {
        const fallback = document.createElement('div');
        fallback.className = 'movie-card__poster--fallback';
        fallback.textContent = '🎬';
        posterWrap.appendChild(fallback);
    }

    if (card.rating) {
        const badge = document.createElement('span');
        badge.className = 'movie-card__rating-badge';
        badge.textContent = `★ ${card.rating.toFixed(1)}`;
        posterWrap.appendChild(badge);
    }

    const info = document.createElement('div');
    info.className = 'movie-card__info';

    const title = document.createElement('h3');
    title.className = 'movie-card__title';
    title.textContent = card.title;

    const meta = document.createElement('div');
    meta.className = 'movie-card__meta';

    if (card.year) {
        const year = document.createElement('span');
        year.className = 'movie-card__year';
        year.textContent = card.year;
        meta.appendChild(year);
    }

    info.appendChild(title);
    info.appendChild(meta);

    a.appendChild(posterWrap);
    a.appendChild(info);

    return a;
}

function scrollToBottom() {
    const container = document.getElementById('chatMessages');
    setTimeout(() => {
        container.scrollTop = container.scrollHeight;
    }, 100);
}

// 初始滚动到底部
document.addEventListener('DOMContentLoaded', function() {
    scrollToBottom();
    document.getElementById('chatInput').focus();
});
