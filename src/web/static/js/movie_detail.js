/**
 * CinemaScope — 电影详情页交互
 * Tab切换、预告片切换
 */
document.addEventListener('DOMContentLoaded', function() {
    // Tab切换
    const tabs = document.querySelectorAll('.detail-tab');
    const contents = document.querySelectorAll('.detail-tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const target = this.dataset.tab;

            // 移除所有激活状态
            tabs.forEach(t => t.classList.remove('detail-tab--active'));
            contents.forEach(c => c.classList.remove('detail-tab-content--active'));

            // 激活当前
            this.classList.add('detail-tab--active');
            const targetContent = document.getElementById('tab-' + target);
            if (targetContent) {
                targetContent.classList.add('detail-tab-content--active');
            }
        });
    });
});

/**
 * 切换详情页预告片视频
 */
function switchTrailer(embedUrl, name) {
    const main = document.querySelector('.trailer-main iframe');
    if (main) {
        main.src = embedUrl;
        // 滚动到预告片区域
        document.getElementById('tab-trailers').scrollIntoView({ behavior: 'smooth' });
    }
}
