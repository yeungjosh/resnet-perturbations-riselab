# ResNet Robustness Explorer - Web Application

An interactive, lightweight web application for exploring ResNet model robustness to input perturbations. Built with vanilla JavaScript, HTML5 Canvas, and Chart.js - no build step required!

## Features

### Interactive Perturbation Playground
- Real-time visualization of Gaussian and Salt & Pepper noise
- Adjustable noise levels with live preview
- Multiple image types (digits, objects, patterns)
- Animated noise effects
- Statistics tracking (corrupted pixels, estimated accuracy)

### Optimizer Comparison Dashboard
- Interactive robustness curves for 5 optimizers (SGD, Adam, Adadelta, Adahessian, Frank-Wolfe)
- Toggle between Gaussian and Salt & Pepper noise types
- Detailed optimizer statistics and descriptions
- Click to show/hide specific optimizers

### Architecture Explorer
- Visual ResNet architecture breakdown
- Training pipeline explanation
- Hyperparameter details
- Key insights into residual learning

### Educational Content
- Research background and methodology
- Why robustness matters in ML
- Dataset information
- Links to papers and resources

## Quick Start

### Local Development

1. **Clone and navigate to webapp directory**:
   ```bash
   cd webapp
   ```

2. **Start a local server**:
   ```bash
   # Python 3
   python3 -m http.server 8000

   # OR Python 2
   python -m SimpleHTTPServer 8000

   # OR Node.js (if you have it)
   npx serve
   ```

3. **Open in browser**:
   ```
   http://localhost:8000
   ```

That's it! No installation, no build step, no dependencies.

## Deploy to Vercel

### One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yeungjosh/resnet-perturbations-riselab&project-name=resnet-robustness-explorer&root-directory=webapp)

### Manual Deploy

1. **Install Vercel CLI** (first time only):
   ```bash
   npm install -g vercel
   ```

2. **Deploy**:
   ```bash
   cd webapp
   vercel
   ```

3. **Follow the prompts**:
   - Set up and deploy: `Y`
   - Which scope: Select your account
   - Link to existing project: `N`
   - Project name: `resnet-robustness-explorer` (or your choice)
   - Directory: `./` (current directory)
   - Override settings: `N`

4. **Production deployment**:
   ```bash
   vercel --prod
   ```

Your app will be live at `https://your-project.vercel.app`!

## Deploy to Other Platforms

### Netlify

1. Drag and drop the `webapp` folder to [Netlify Drop](https://app.netlify.com/drop)
2. Done! Your site is live.

### GitHub Pages

1. Push the `webapp` directory to your GitHub repo
2. Go to Settings > Pages
3. Select source: `main` branch, `/webapp` folder
4. Your site will be at `https://yourusername.github.io/resnet-perturbations-riselab`

### Any Static Host

Simply upload the contents of the `webapp` directory to any static file host:
- AWS S3 + CloudFront
- Google Cloud Storage
- Azure Static Web Apps
- Cloudflare Pages
- Surge.sh
- Render

## Project Structure

```
webapp/
├── index.html          # Main HTML structure
├── app.js              # JavaScript functionality
├── vercel.json         # Vercel configuration
├── package.json        # Project metadata
└── README.md           # This file
```

## Technology Stack

- **HTML5 Canvas**: For interactive perturbation visualizations
- **Chart.js 4.4.0**: For robustness curve plotting
- **Vanilla JavaScript**: No frameworks, no build tools
- **CSS3**: Modern gradients, animations, and responsive design
- **Google Fonts**: Inter font family

## Browser Compatibility

Works on all modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Customization

### Change Colors

Edit CSS variables in `index.html`:
```css
:root {
    --primary: #6366f1;        /* Main accent color */
    --secondary: #ec4899;       /* Secondary accent */
    --bg-dark: #0f172a;        /* Background */
    --bg-card: #1e293b;        /* Card background */
}
```

### Add New Optimizers

In `app.js`, add data to the `datasets` object:
```javascript
{
    label: 'NewOptimizer',
    data: [98.0, 97.5, ...],  // Accuracy at each noise level
    borderColor: '#your-color',
    // ... other Chart.js options
}
```

### Modify Perturbation Types

Add new perturbation functions in `app.js`:
```javascript
function applyCustomNoise(canvas, originalData, noiseLevel) {
    // Your custom noise implementation
}
```

## Performance

- **Lightweight**: ~50KB total (HTML + JS)
- **Fast**: No build step, instant loading
- **Responsive**: Works on mobile and desktop
- **Accessible**: Keyboard navigation, semantic HTML

## Development

### File Watching (Optional)

For live reload during development, use:
```bash
# With browser-sync (requires npm)
npx browser-sync start --server --files "*.html, *.js, *.css"

# With live-server
npx live-server
```

### Debugging

Open browser DevTools console to see:
- Performance metrics
- Interactive state
- Chart data
- Easter eggs 🎉

## Features Showcase

### 1. Interactive Perturbation Visualization
Real-time canvas rendering shows how noise affects images. Adjust the slider to see immediate changes in both Gaussian and Salt & Pepper noise patterns.

### 2. Robustness Curves
Based on real ResNet training experiments, the curves show how different optimizers produce models with varying noise resilience. Click optimizer cards to show/hide specific lines.

### 3. Architecture Visualization
Understand how ResNet's skip connections enable deep networks and why this architecture is chosen for robustness studies.

### 4. Responsive Design
Works beautifully on phones, tablets, and desktops with a dark theme optimized for long reading sessions.

## Educational Use

Perfect for:
- Teaching ML robustness concepts
- Conference presentations
- Research demos
- Student projects
- Understanding optimizer differences

## Contributing

This webapp is part of the larger ResNet Perturbations research project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes in the `webapp` directory
4. Test locally
5. Submit a pull request

## Research Context

This visualization tool accompanies the research paper on optimizer robustness. For the full study, training code, and analysis notebooks, see the main repository:

[github.com/yeungjosh/resnet-perturbations-riselab](https://github.com/yeungjosh/resnet-perturbations-riselab)

## License

MIT License - See parent repository for details

## Credits

- **Research**: Mahoney Group, RISELab @ UC Berkeley
- **Visualization**: Built with Chart.js
- **Design**: Inter font family by Rasmus Andersson
- **Icons**: Unicode emoji

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: RISELab @ UC Berkeley
- Documentation: See main repository README

---

**Built with ❤️ for reproducible ML research**

Enjoy exploring neural network robustness! 🚀
