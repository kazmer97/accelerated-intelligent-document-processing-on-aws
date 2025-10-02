# Running the IDP UI Locally for Development

## Prerequisites

1. Node.js and npm installed
2. AWS credentials configured (for backend API access)
3. The IDP stack deployed in AWS

## Setup Steps

### 1. Install Dependencies

```bash
cd /Users/kaznb/dev/accelerated-intelligent-document-processing-on-aws/src/ui
npm install

# If you encounter dependency issues, you can also try:
npm install --legacy-peer-deps
```

### 2. Configure AWS Exports

The UI needs to connect to your deployed backend. Check if `src/aws-exports.js`
exists and is configured properly. If not, you'll need to create it with your
stack outputs.

### 3. Start the Development Server

```bash
npm start
```

This will start the React development server on http://localhost:3000

### 4. Enable Hot Module Replacement

The development server already includes hot module replacement, so any changes
you make to the code will automatically refresh in the browser.

## Tracking Changes

### Watch for File Changes

The React development server automatically watches for file changes in the
`src/` directory.

### Browser Developer Tools

1. Open Chrome DevTools (F12 or right-click → Inspect)
2. Go to the "Sources" tab to debug JavaScript
3. Use the "Network" tab to monitor API calls
4. Check the "Console" for logs and errors

### React Developer Tools

Install the React Developer Tools browser extension for better debugging:

- [Chrome Extension](https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi)
- [Firefox Extension](https://addons.mozilla.org/en-US/firefox/addon/react-devtools/)

### Useful Development Commands

```bash
# Run in the UI directory

# Check for linting issues
npm run lint

# Run tests
npm test

# Build for production (to test the production build)
npm run build
```

## Testing the New Schema Builder

Once the development server is running, you can access the new schema builder
at:

- http://localhost:3000/configuration (once we integrate it)
- Or directly at http://localhost:3000/schema-builder (standalone route we'll
  create)

## Common Issues and Solutions

### Issue: CORS errors when calling AWS services

**Solution**: Make sure your Cognito and API Gateway are configured to allow
localhost:3000 as an origin.

### Issue: Authentication errors

**Solution**: You may need to sign in through the UI first, or configure mock
authentication for local development.

### Issue: Module not found errors

**Solution**: Clear node_modules and reinstall:

```bash
rm -rf node_modules package-lock.json
npm install
```

### Issue: Port 3000 already in use

**Solution**: Either kill the process using port 3000 or use a different port:

```bash
PORT=3001 npm start
```
