/**
 * CinemaScope — 全局交互脚本
 * 导航栏、搜索、卡片效果等
 */
document.addEventListener('DOMContentLoaded', function() {
    // 导航栏滚动阴影
    const navbar = document.getElementById('navbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 10) {
                navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.4)';
            } else {
                navbar.style.boxShadow = 'none';
            }
        });
    }

    // 所有电影卡片的hover音效（微妙的光晕效果用CSS处理）
});
