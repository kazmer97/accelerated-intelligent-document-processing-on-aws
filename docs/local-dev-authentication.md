# Setting Up Authentication for Local Development

## Step 1: Get Your Stack Outputs

First, you need to get the AWS stack outputs from your deployed IDP solution.
You have two options:

### Option A: Using AWS CLI

```bash
# Replace YOUR_STACK_NAME with your actual CloudFormation stack name
aws cloudformation describe-stacks --stack-name YOUR_STACK_NAME --query 'Stacks[0].Outputs' --output table

# Or get specific values:
aws cloudformation describe-stacks --stack-name YOUR_STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`UserPoolId`].OutputValue' --output text
aws cloudformation describe-stacks --stack-name YOUR_STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`UserPoolClientId`].OutputValue' --output text
aws cloudformation describe-stacks --stack-name YOUR_STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`IdentityPoolId`].OutputValue' --output text
aws cloudformation describe-stacks --stack-name YOUR_STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`GraphQLApiEndpoint`].OutputValue' --output text
```

### Option B: Using AWS Console

1. Go to AWS CloudFormation Console
2. Find your IDP stack
3. Click on the "Outputs" tab
4. Copy the values for:
   - UserPoolId
   - UserPoolClientId
   - IdentityPoolId
   - GraphQLApiEndpoint
   - AWS Region (from the console URL)

## Step 2: Create Environment File

Create a `.env.local` file in the UI directory:

```bash
cd /Users/kaznb/dev/accelerated-intelligent-document-processing-on-aws/src/ui
```

Create `.env.local` with the following content (replace with your actual
values):

```env
# AWS Configuration
REACT_APP_AWS_REGION=us-east-1
REACT_APP_USER_POOL_ID=us-east-1_XXXXXXXXX
REACT_APP_USER_POOL_CLIENT_ID=XXXXXXXXXXXXXXXXXXXXXXXXXX
REACT_APP_IDENTITY_POOL_ID=us-east-1:XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
REACT_APP_APPSYNC_GRAPHQL_URL=https://XXXXXXXXXXXXXXXXXXXXXXXXXX.appsync-api.us-east-1.amazonaws.com/graphql
```

## Step 3: Start the Development Server

```bash
# Install dependencies if not already done
npm install

# Start the development server
npm start
```

The app will open at http://localhost:3000

## Step 4: Login Process

### First-Time User Setup

1. When the app opens, you'll see a login screen
2. Click "Create Account" if you don't have one
3. Enter:
   - Username
   - Email address
   - Password (minimum 8 characters)
4. Check your email for the verification code
5. Enter the verification code
6. You'll be logged in automatically

### Existing User Login

1. Enter your username/email
2. Enter your password
3. Click "Sign In"

### Alternative: Create User via AWS CLI

```bash
# Create a user
aws cognito-idp admin-create-user \
  --user-pool-id YOUR_USER_POOL_ID \
  --username testuser \
  --user-attributes Name=email,Value=your-email@example.com \
  --message-action SUPPRESS \
  --temporary-password TempPassword123!

# Set permanent password
aws cognito-idp admin-set-user-password \
  --user-pool-id YOUR_USER_POOL_ID \
  --username testuser \
  --password YourPassword123! \
  --permanent
```

## Step 5: Troubleshooting

### Issue: CORS errors

If you see CORS errors in the browser console:

1. Check API Gateway CORS settings:

```bash
# Get the API ID
API_ID=$(aws cloudformation describe-stacks --stack-name YOUR_STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`GraphQLApiId`].OutputValue' --output text)

# You may need to add localhost:3000 to allowed origins in AppSync console
```

2. Update Cognito App Client settings to allow localhost:

```bash
aws cognito-idp update-user-pool-client \
  --user-pool-id YOUR_USER_POOL_ID \
  --client-id YOUR_CLIENT_ID \
  --allowed-o-auth-flows-user-pool-client \
  --allowed-o-auth-flows code \
  --allowed-o-auth-scopes openid email profile \
  --callback-urls http://localhost:3000 \
  --logout-urls http://localhost:3000
```

### Issue: Authentication not working

Check browser console for errors and verify:

1. All environment variables are set correctly
2. The stack is deployed and running
3. Your AWS credentials have access to the resources

### Issue: GraphQL errors

Ensure your user has the correct IAM permissions through Cognito Identity Pool.

## Step 6: Quick Test

Once logged in, you should be able to:

1. Navigate to the Configuration page
2. View document lists
3. Access all UI features

## Optional: Mock Authentication for Development

If you want to bypass authentication during development, you can create a mock
auth context. However, this won't allow you to access real AWS resources.

Create `src/hooks/use-mock-auth.js`:

```javascript
export const useMockAuth = () => ({
  user: { username: 'dev-user' },
  signIn: async () => true,
  signOut: async () => true,
  isAuthenticated: true,
});
```

Then conditionally use it in development mode.
