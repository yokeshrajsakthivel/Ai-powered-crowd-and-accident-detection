function startCrowd() {
    fetch('/start-crowd', { method: 'POST' })
    .then(response => response.json())
    .then(data => alert(data.message))
    .catch(error => console.error(error));
}

function startTraffic() {
    const videoPath = prompt("Enter full path of video file:");
    if (!videoPath) return;

    fetch('/start-traffic', { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_file: videoPath })
    })
    .then(response => response.json())
    .then(data => alert(data.message))
    .catch(error => console.error(error));
}

// Function to toggle the sidebar
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('active');
}

    